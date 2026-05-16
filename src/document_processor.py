"""Document processing for insurance application extraction.

Uses Amazon Bedrock for document understanding to extract structured
data from uploaded insurance applications (PDF, images).
"""

import io
import json
import logging
from typing import Optional

from src.bedrock_client import BedrockClient, GuardrailIntervention
from src.models import (
    ApplicationData,
    AutoDetails,
    CommercialDetails,
    LifeDetails,
    LineOfBusiness,
    PropertyDetails,
)
from src.prompts import DOCUMENT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# Supported document formats
SUPPORTED_FORMATS = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "text/plain": "txt",
}


class DocumentProcessor:
    """Process uploaded insurance application documents.

    Extracts structured data from PDFs and images using Bedrock's
    multimodal capabilities with guardrails applied.
    """

    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        self.bedrock_client = bedrock_client

    def process_document(
        self,
        file_bytes: bytes,
        file_type: str,
        file_name: str = "document",
    ) -> dict:
        """Process an uploaded document and extract application data.

        Args:
            file_bytes: Raw file content.
            file_type: MIME type of the uploaded file.
            file_name: Original filename for context.

        Returns:
            Dict with 'application_data' (ApplicationData) and 'raw_extraction' (str).
        """
        # Validate file type
        doc_format = SUPPORTED_FORMATS.get(file_type)
        if not doc_format:
            return {
                "error": f"Unsupported file type: {file_type}. "
                         f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}",
                "application_data": None,
                "raw_extraction": None,
            }

        # Handle text files differently
        if doc_format == "txt":
            return self._process_text_document(file_bytes, file_name)

        # Use Bedrock for PDF/image understanding
        return self._process_visual_document(file_bytes, doc_format, file_name)

    def _process_visual_document(
        self,
        file_bytes: bytes,
        doc_format: str,
        file_name: str,
    ) -> dict:
        """Process PDF or image documents using Bedrock multimodal."""
        if not self.bedrock_client:
            return {
                "error": "Bedrock client not configured. Cannot process visual documents.",
                "application_data": None,
                "raw_extraction": None,
            }

        prompt = DOCUMENT_EXTRACTION_PROMPT.format(
            document_content="[See attached document]"
        )

        try:
            response = self.bedrock_client.converse_with_document(
                document_bytes=file_bytes,
                document_format=doc_format,
                prompt=prompt,
                system_prompt=(
                    "You are an insurance document processing specialist. "
                    "Extract all relevant underwriting data from the provided document. "
                    "Return structured JSON that can be parsed into application fields."
                ),
            )

            raw_extraction = response["content"]
            application_data = self._parse_extraction(raw_extraction)

            return {
                "application_data": application_data,
                "raw_extraction": raw_extraction,
                "error": None,
            }

        except GuardrailIntervention as e:
            logger.warning("Guardrail blocked document processing: %s", str(e))
            return {
                "error": "Document processing was filtered by safety controls. "
                         "The document may contain content outside underwriting scope.",
                "application_data": None,
                "raw_extraction": None,
            }
        except Exception as e:
            logger.error("Document processing failed: %s", str(e))
            return {
                "error": f"Document processing failed: {str(e)}",
                "application_data": None,
                "raw_extraction": None,
            }

    def _process_text_document(self, file_bytes: bytes, file_name: str) -> dict:
        """Process plain text documents."""
        try:
            text_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "error": "Unable to decode text file. Please ensure UTF-8 encoding.",
                "application_data": None,
                "raw_extraction": None,
            }

        if not self.bedrock_client:
            # Without LLM, attempt basic parsing
            return {
                "application_data": None,
                "raw_extraction": text_content[:5000],
                "error": "Bedrock client not configured. Raw text extracted only.",
            }

        prompt = DOCUMENT_EXTRACTION_PROMPT.format(document_content=text_content[:10000])

        try:
            response = self.bedrock_client.converse(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are an insurance document processing specialist. "
                    "Extract all relevant underwriting data from the text. "
                    "Return structured JSON."
                ),
                temperature=0.1,
            )

            raw_extraction = response["content"]
            application_data = self._parse_extraction(raw_extraction)

            return {
                "application_data": application_data,
                "raw_extraction": raw_extraction,
                "error": None,
            }
        except GuardrailIntervention as e:
            return {
                "error": str(e),
                "application_data": None,
                "raw_extraction": text_content[:2000],
            }
        except Exception as e:
            return {
                "error": f"Text processing failed: {str(e)}",
                "application_data": None,
                "raw_extraction": text_content[:2000],
            }

    def _parse_extraction(self, raw_text: str) -> Optional[ApplicationData]:
        """Parse LLM extraction output into ApplicationData model.

        Attempts to find and parse JSON from the LLM response.
        """
        # Try to find JSON in the response
        json_str = self._extract_json(raw_text)
        if not json_str:
            logger.warning("No JSON found in extraction output")
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from extraction")
            return None

        # Map extracted fields to ApplicationData
        try:
            lob = self._detect_line_of_business(data)
            application = ApplicationData(
                applicant_name=data.get("applicant_name", ""),
                line_of_business=lob,
                requested_coverage_limit=data.get("coverage_limit"),
                requested_deductible=data.get("deductible"),
                notes=data.get("notes", ""),
            )

            # Populate line-specific details
            if lob == LineOfBusiness.PROPERTY:
                application.property_details = PropertyDetails(
                    address=data.get("address", ""),
                    construction_type=data.get("construction_type", ""),
                    year_built=data.get("year_built"),
                    square_footage=data.get("square_footage"),
                    prior_claims=data.get("prior_claims", 0),
                    occupancy_type=data.get("occupancy_type", ""),
                )
            elif lob == LineOfBusiness.AUTO:
                application.auto_details = AutoDetails(
                    vehicle_year=data.get("vehicle_year"),
                    vehicle_make=data.get("vehicle_make", ""),
                    vehicle_model=data.get("vehicle_model", ""),
                    driver_age=data.get("driver_age"),
                    annual_mileage=data.get("annual_mileage"),
                    prior_accidents=data.get("prior_accidents", 0),
                )
            elif lob == LineOfBusiness.LIFE:
                application.life_details = LifeDetails(
                    applicant_age=data.get("applicant_age"),
                    smoker=data.get("smoker", False),
                    face_amount=data.get("face_amount"),
                    term_years=data.get("term_years"),
                )
            elif lob == LineOfBusiness.COMMERCIAL:
                application.commercial_details = CommercialDetails(
                    business_name=data.get("business_name", ""),
                    naics_code=data.get("naics_code", ""),
                    years_in_business=data.get("years_in_business"),
                    annual_revenue=data.get("annual_revenue"),
                    employee_count=data.get("employee_count"),
                    prior_losses=data.get("prior_losses", 0),
                )

            return application

        except Exception as e:
            logger.error("Failed to build ApplicationData from extraction: %s", str(e))
            return None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON block from LLM text output."""
        # Try to find ```json blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()

        # Try to find raw JSON object
        brace_start = text.find("{")
        if brace_start == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start:i + 1]

        return None

    def _detect_line_of_business(self, data: dict) -> LineOfBusiness:
        """Detect line of business from extracted data fields."""
        lob_str = data.get("line_of_business", "").lower()

        if lob_str in ("property", "homeowners", "dwelling", "ho"):
            return LineOfBusiness.PROPERTY
        elif lob_str in ("auto", "automobile", "vehicle", "car"):
            return LineOfBusiness.AUTO
        elif lob_str in ("life", "term life", "whole life"):
            return LineOfBusiness.LIFE
        elif lob_str in ("commercial", "business", "gl", "general liability"):
            return LineOfBusiness.COMMERCIAL

        # Infer from available fields
        if any(k in data for k in ("vehicle_make", "driver_age", "vin")):
            return LineOfBusiness.AUTO
        elif any(k in data for k in ("face_amount", "smoker", "beneficiary")):
            return LineOfBusiness.LIFE
        elif any(k in data for k in ("naics_code", "business_name", "employee_count")):
            return LineOfBusiness.COMMERCIAL
        else:
            return LineOfBusiness.PROPERTY
