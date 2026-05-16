"""Amazon Bedrock client with Guardrails integration.

Handles model inference through the Bedrock Converse API with content filtering,
PII redaction, and topic denial controls applied transparently.
"""

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.guardrails_config import GuardrailsConfig, build_guardrail_params, get_guardrails_config

logger = logging.getLogger(__name__)


class GuardrailIntervention(Exception):
    """Raised when Bedrock Guardrails blocks or modifies a response."""

    def __init__(self, message: str, trace: Optional[dict] = None):
        super().__init__(message)
        self.trace = trace


class BedrockClient:
    """Client for Amazon Bedrock inference with Guardrails.

    Wraps the Bedrock Converse API to provide:
    - Model inference with Claude 3 Sonnet
    - Automatic Guardrails application (content filtering, PII, topic denial)
    - Structured error handling for guardrail interventions
    - Token usage tracking
    """

    MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
    MAX_TOKENS = 4096

    def __init__(
        self,
        region: str = "us-west-2",
        guardrails_config: Optional[GuardrailsConfig] = None,
    ):
        self.region = region
        self.guardrails_config = guardrails_config or get_guardrails_config()
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    def converse(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = MAX_TOKENS,
    ) -> dict:
        """Send a conversation to Bedrock with Guardrails applied.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: System-level instructions for the model.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            Dict with 'content', 'stop_reason', 'usage', and 'guardrail_trace' keys.

        Raises:
            GuardrailIntervention: When guardrails block the input or output.
            ClientError: For AWS API errors.
        """
        # Build the API request
        request_params = {
            "modelId": self.MODEL_ID,
            "messages": self._format_messages(messages),
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        # Add system prompt
        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        # Add guardrail configuration
        guardrail_params = build_guardrail_params(self.guardrails_config)
        if guardrail_params:
            request_params["guardrailConfig"] = guardrail_params

        try:
            response = self.client.converse(**request_params)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                logger.warning("Bedrock throttling encountered, consider exponential backoff")
            raise

        # Process the response
        result = self._process_response(response)

        # Check for guardrail interventions
        stop_reason = response.get("stopReason", "")
        if stop_reason == "guardrail_intervened":
            trace = response.get("trace", {}).get("guardrail", {})
            logger.warning(
                "Guardrail intervention: %s",
                json.dumps(trace, default=str)[:500],
            )
            raise GuardrailIntervention(
                message=self.guardrails_config.blocked_output_message,
                trace=trace,
            )

        return result

    def converse_with_document(
        self,
        document_bytes: bytes,
        document_format: str,
        prompt: str,
        system_prompt: str = "",
    ) -> dict:
        """Send a document to Bedrock for understanding/extraction.

        Args:
            document_bytes: Raw document bytes.
            document_format: Format string (pdf, png, jpeg, etc.).
            prompt: Instructions for processing the document.
            system_prompt: System-level instructions.

        Returns:
            Dict with extracted content and metadata.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": document_format,
                            "name": "application_document",
                            "source": {"bytes": document_bytes},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ]

        request_params = {
            "modelId": self.MODEL_ID,
            "messages": messages,
            "inferenceConfig": {
                "temperature": 0.1,  # Low temp for extraction accuracy
                "maxTokens": self.MAX_TOKENS,
            },
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        guardrail_params = build_guardrail_params(self.guardrails_config)
        if guardrail_params:
            request_params["guardrailConfig"] = guardrail_params

        try:
            response = self.client.converse(**request_params)
        except ClientError as e:
            logger.error("Document processing failed: %s", str(e))
            raise

        return self._process_response(response)

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        """Format messages for the Bedrock Converse API.

        Converts simple {'role': 'user', 'content': 'text'} format
        to the Bedrock Converse API format.
        """
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                formatted.append({
                    "role": role,
                    "content": [{"text": content}],
                })
            elif isinstance(content, list):
                # Already in Bedrock format
                formatted.append({"role": role, "content": content})
            else:
                formatted.append({
                    "role": role,
                    "content": [{"text": str(content)}],
                })

        return formatted

    def _process_response(self, response: dict) -> dict:
        """Extract content and metadata from Bedrock response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        # Extract text content
        text_content = ""
        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]

        # Track token usage
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        return {
            "content": text_content,
            "stop_reason": response.get("stopReason", ""),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "guardrail_trace": response.get("trace", {}).get("guardrail"),
        }

    def check_guardrail_health(self) -> dict:
        """Verify that the configured guardrail is accessible and active.

        Returns:
            Dict with 'healthy' bool and 'details' string.
        """
        if not self.guardrails_config.guardrail_id:
            return {
                "healthy": False,
                "details": "No guardrail ID configured. Set BEDROCK_GUARDRAIL_ID env var.",
            }

        try:
            bedrock_client = boto3.client("bedrock", region_name=self.region)
            response = bedrock_client.get_guardrail(
                guardrailIdentifier=self.guardrails_config.guardrail_id,
                guardrailVersion=self.guardrails_config.guardrail_version,
            )
            status = response.get("status", "UNKNOWN")
            return {
                "healthy": status == "READY",
                "details": f"Guardrail status: {status}",
            }
        except ClientError as e:
            return {
                "healthy": False,
                "details": f"Guardrail check failed: {str(e)}",
            }
