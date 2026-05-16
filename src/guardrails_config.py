"""Configuration for Amazon Bedrock Guardrails.

Defines content filtering, topic denial, PII handling, and grounding
controls for responsible AI in the underwriting domain.
"""

import os
from dataclasses import dataclass, field


@dataclass
class GuardrailsConfig:
    """Bedrock Guardrails configuration for underwriting assistant."""

    guardrail_id: str = field(
        default_factory=lambda: os.environ.get("BEDROCK_GUARDRAIL_ID", "")
    )
    guardrail_version: str = field(
        default_factory=lambda: os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
    )

    # Denied topics - the model must not provide information on these
    denied_topics: list[dict] = field(default_factory=lambda: [
        {
            "name": "MedicalDiagnosis",
            "definition": "Providing specific medical diagnoses, treatment recommendations, "
                         "or health prognoses for insurance applicants",
            "examples": [
                "Based on these symptoms, the applicant likely has diabetes",
                "This applicant's condition suggests a life expectancy of",
                "The applicant should seek treatment for their cardiac condition",
            ],
            "type": "DENY",
        },
        {
            "name": "CompetitorInformation",
            "definition": "Providing specific pricing, coverage details, or recommendations "
                         "about competitor insurance products or companies",
            "examples": [
                "State Farm offers a better rate for this risk profile",
                "Compared to Allstate's policy, you should",
                "The competitor's combined ratio suggests their pricing is",
            ],
            "type": "DENY",
        },
        {
            "name": "DiscriminatoryFactors",
            "definition": "Using protected class characteristics (race, religion, national origin, "
                         "genetic information) as underwriting factors",
            "examples": [
                "Applicants from this neighborhood tend to have higher claims",
                "Based on their ethnic background, the risk profile suggests",
                "Genetic test results indicate higher mortality risk",
            ],
            "type": "DENY",
        },
        {
            "name": "LegalAdvice",
            "definition": "Providing specific legal advice about insurance coverage disputes, "
                         "regulatory compliance, or litigation strategy",
            "examples": [
                "You should sue the policyholder for fraud",
                "This claim denial will hold up in court because",
                "The regulatory filing should state",
            ],
            "type": "DENY",
        },
    ])

    # Content filters with strength levels
    content_filters: dict = field(default_factory=lambda: {
        "hate": {"input_strength": "HIGH", "output_strength": "HIGH"},
        "insults": {"input_strength": "HIGH", "output_strength": "HIGH"},
        "sexual": {"input_strength": "HIGH", "output_strength": "HIGH"},
        "violence": {"input_strength": "HIGH", "output_strength": "HIGH"},
        "misconduct": {"input_strength": "MEDIUM", "output_strength": "HIGH"},
        "prompt_attack": {"input_strength": "HIGH", "output_strength": "NONE"},
    })

    # PII handling configuration
    pii_config: dict = field(default_factory=lambda: {
        "action": "ANONYMIZE",  # ANONYMIZE or BLOCK
        "entities": [
            {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "ANONYMIZE"},
            {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "ANONYMIZE"},
            {"type": "US_BANK_ACCOUNT_NUMBER", "action": "ANONYMIZE"},
            {"type": "US_BANK_ROUTING_NUMBER", "action": "ANONYMIZE"},
            {"type": "EMAIL", "action": "ANONYMIZE"},
            {"type": "PHONE", "action": "ANONYMIZE"},
            {"type": "US_PASSPORT_NUMBER", "action": "ANONYMIZE"},
            {"type": "DRIVER_ID", "action": "ANONYMIZE"},
            {"type": "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "action": "ANONYMIZE"},
        ],
    })

    # Contextual grounding configuration
    grounding_config: dict = field(default_factory=lambda: {
        "grounding_threshold": 0.7,
        "relevance_threshold": 0.7,
    })

    # Blocked messaging
    blocked_input_message: str = (
        "I'm unable to process this request. The input contains content that falls "
        "outside my underwriting scope. Please rephrase your question to focus on "
        "risk assessment, coverage recommendations, or application review."
    )
    blocked_output_message: str = (
        "I've generated a response that was filtered by our safety controls. "
        "This typically happens when the response would include medical advice, "
        "discriminatory factors, or competitor-specific information. "
        "Please ask a more specific underwriting question."
    )


def get_guardrails_config() -> GuardrailsConfig:
    """Factory function to create guardrails configuration."""
    return GuardrailsConfig()


def build_guardrail_params(config: GuardrailsConfig) -> dict:
    """Build the guardrail parameters for Bedrock API calls.

    Returns the guardrailIdentifier and guardrailVersion for use in
    Bedrock Converse or InvokeModel API calls.
    """
    if not config.guardrail_id:
        return {}

    return {
        "guardrailIdentifier": config.guardrail_id,
        "guardrailVersion": config.guardrail_version,
    }
