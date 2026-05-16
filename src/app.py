"""Streamlit application for the GenAI Underwriting Assistant.

Provides a professional interface for insurance underwriting with:
- Chat interface for underwriting queries
- Structured risk assessment display
- Document upload for application review
- Configurable risk appetite and line of business
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bedrock_client import BedrockClient, GuardrailIntervention
from src.document_processor import DocumentProcessor
from src.models import (
    ApplicationData,
    AutoDetails,
    CommercialDetails,
    LifeDetails,
    LineOfBusiness,
    PropertyDetails,
    RiskAppetite,
    RiskCategory,
)
from src.prompts import CHAT_RESPONSE_PROMPT, SYSTEM_PROMPT
from src.underwriting_engine import UnderwritingEngine

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="GenAI Underwriting Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .risk-score-low { color: #27AE60; font-size: 2rem; font-weight: bold; }
    .risk-score-medium { color: #F39C12; font-size: 2rem; font-weight: bold; }
    .risk-score-high { color: #E74C3C; font-size: 2rem; font-weight: bold; }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1B4F72;
    }
    .stAlert > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_application" not in st.session_state:
        st.session_state.current_application = None
    if "risk_assessment" not in st.session_state:
        st.session_state.risk_assessment = None
    if "underwriting_decision" not in st.session_state:
        st.session_state.underwriting_decision = None
    if "bedrock_client" not in st.session_state:
        try:
            st.session_state.bedrock_client = BedrockClient()
        except Exception:
            st.session_state.bedrock_client = None
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor(
            bedrock_client=st.session_state.bedrock_client
        )


init_session_state()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("Configuration")
    st.divider()

    # Line of Business Selection
    lob = st.selectbox(
        "Line of Business",
        options=[lob.value for lob in LineOfBusiness],
        format_func=lambda x: x.replace("_", " ").title(),
        index=0,
    )
    selected_lob = LineOfBusiness(lob)

    # Risk Appetite
    appetite = st.selectbox(
        "Risk Appetite",
        options=[ra.value for ra in RiskAppetite],
        format_func=lambda x: x.title(),
        index=1,  # Default to moderate
    )
    selected_appetite = RiskAppetite(appetite)

    st.divider()

    # Guardrails Status
    st.subheader("Safety Controls")
    if st.session_state.bedrock_client:
        health = st.session_state.bedrock_client.check_guardrail_health()
        if health["healthy"]:
            st.success("Guardrails: Active")
        else:
            st.warning(f"Guardrails: {health['details']}")
    else:
        st.info("Bedrock client not connected")

    st.caption("Content filtering: Enabled")
    st.caption("PII redaction: Enabled")
    st.caption("Topic denial: Medical, Legal, Competitor")

    st.divider()

    # Token Usage
    st.subheader("Session Metrics")
    if st.session_state.bedrock_client:
        st.metric("Input Tokens", st.session_state.bedrock_client.total_input_tokens)
        st.metric("Output Tokens", st.session_state.bedrock_client.total_output_tokens)

    # Reset button
    st.divider()
    if st.button("Reset Session", type="secondary", use_container_width=True):
        for key in ["messages", "current_application", "risk_assessment", "underwriting_decision"]:
            st.session_state[key] = None if key != "messages" else []
        st.rerun()

# -----------------------------------------------------------------------------
# Main Content Area
# -----------------------------------------------------------------------------

st.title("GenAI Underwriting Assistant")
st.caption("AI-powered risk assessment with Amazon Bedrock Guardrails")

# Tab layout
tab_chat, tab_application, tab_document = st.tabs([
    "Chat Interface",
    "Application Entry",
    "Document Upload",
])

# -----------------------------------------------------------------------------
# Tab 1: Chat Interface
# -----------------------------------------------------------------------------

with tab_chat:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about underwriting, risk assessment, or coverage..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response_text = _handle_chat_message(prompt, selected_lob, selected_appetite)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# -----------------------------------------------------------------------------
# Tab 2: Application Entry
# -----------------------------------------------------------------------------

with tab_application:
    st.subheader(f"New {selected_lob.value.title()} Application")

    with st.form("application_form"):
        col1, col2 = st.columns(2)

        with col1:
            applicant_name = st.text_input("Applicant Name")
            coverage_limit = st.number_input(
                "Requested Coverage Limit ($)",
                min_value=10000,
                max_value=50000000,
                value=500000,
                step=50000,
            )

        with col2:
            deductible = st.number_input(
                "Requested Deductible ($)",
                min_value=0,
                max_value=100000,
                value=2500,
                step=500,
            )

        st.divider()

        # Line-specific fields
        application_data = _render_lob_fields(selected_lob)

        # Submit button
        submitted = st.form_submit_button(
            "Run Underwriting Assessment",
            type="primary",
            use_container_width=True,
        )

        if submitted and applicant_name:
            application = ApplicationData(
                applicant_name=applicant_name,
                line_of_business=selected_lob,
                requested_coverage_limit=float(coverage_limit),
                requested_deductible=float(deductible),
                **application_data,
            )
            st.session_state.current_application = application

            # Run assessment
            engine = UnderwritingEngine(
                bedrock_client=st.session_state.bedrock_client,
                risk_appetite=selected_appetite,
            )

            with st.spinner("Performing risk assessment..."):
                assessment = engine.assess_risk(application)
                st.session_state.risk_assessment = assessment

                coverage = engine.recommend_coverage(application, assessment)
                decision = engine.make_decision(application, assessment, coverage)
                st.session_state.underwriting_decision = decision

            st.success("Assessment complete!")
            st.rerun()

    # Display results if available
    if st.session_state.underwriting_decision:
        _display_underwriting_results()

# -----------------------------------------------------------------------------
# Tab 3: Document Upload
# -----------------------------------------------------------------------------

with tab_document:
    st.subheader("Upload Insurance Application Document")
    st.caption("Supported formats: PDF, PNG, JPEG, TXT")

    uploaded_file = st.file_uploader(
        "Upload application document",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        help="Upload an insurance application form for automated data extraction",
    )

    if uploaded_file:
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Extract Application Data", type="primary"):
            with st.spinner("Processing document with Bedrock..."):
                processor = st.session_state.document_processor
                result = processor.process_document(
                    file_bytes=uploaded_file.getvalue(),
                    file_type=uploaded_file.type,
                    file_name=uploaded_file.name,
                )

            if result.get("error"):
                st.error(result["error"])

            if result.get("raw_extraction"):
                with st.expander("Raw Extraction Output", expanded=False):
                    st.code(result["raw_extraction"], language="json")

            if result.get("application_data"):
                st.success("Application data extracted successfully!")
                st.session_state.current_application = result["application_data"]

                with st.expander("Extracted Application Data", expanded=True):
                    st.json(result["application_data"].model_dump(mode="json"))

                if st.button("Run Assessment on Extracted Data"):
                    engine = UnderwritingEngine(
                        bedrock_client=st.session_state.bedrock_client,
                        risk_appetite=selected_appetite,
                    )
                    assessment = engine.assess_risk(result["application_data"])
                    coverage = engine.recommend_coverage(result["application_data"], assessment)
                    decision = engine.make_decision(
                        result["application_data"], assessment, coverage
                    )
                    st.session_state.risk_assessment = assessment
                    st.session_state.underwriting_decision = decision
                    st.rerun()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _handle_chat_message(
    message: str,
    lob: LineOfBusiness,
    appetite: RiskAppetite,
) -> str:
    """Process a chat message and generate response."""
    client = st.session_state.bedrock_client
    if not client:
        return (
            "Bedrock client is not configured. Please set AWS credentials and "
            "ensure Amazon Bedrock access is enabled in your account."
        )

    # Build context from current application
    app_context = "No application currently loaded."
    if st.session_state.current_application:
        app_context = st.session_state.current_application.model_dump_json(indent=2)

    prompt = CHAT_RESPONSE_PROMPT.format(
        application_context=app_context,
        question=message,
    )

    # Build conversation history
    history = []
    for msg in st.session_state.messages[-10:]:  # Last 10 messages for context
        history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": prompt})

    try:
        response = client.converse(
            messages=history,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
        )
        return response["content"]
    except GuardrailIntervention as e:
        return (
            "**Safety Control Activated**\n\n"
            f"{str(e)}\n\n"
            "This typically occurs when a question involves medical diagnoses, "
            "competitor information, or discriminatory factors. "
            "Please rephrase your question to focus on underwriting risk assessment."
        )
    except Exception as e:
        return f"Error generating response: {str(e)}"


def _render_lob_fields(lob: LineOfBusiness) -> dict:
    """Render line-of-business specific form fields and return data dict."""
    if lob == LineOfBusiness.PROPERTY:
        col1, col2 = st.columns(2)
        with col1:
            address = st.text_input("Property Address")
            construction = st.selectbox(
                "Construction Type",
                ["frame", "masonry", "fire-resistive"],
            )
            year_built = st.number_input("Year Built", 1900, 2026, 2000)
        with col2:
            protection_class = st.slider("Protection Class (1-10)", 1, 10, 5)
            prior_claims = st.number_input("Prior Claims (5 years)", 0, 20, 0)
            occupancy = st.selectbox(
                "Occupancy Type",
                ["owner-occupied", "tenant", "vacant"],
            )
        return {
            "property_details": PropertyDetails(
                address=address,
                construction_type=construction,
                year_built=year_built,
                protection_class=protection_class,
                prior_claims=prior_claims,
                occupancy_type=occupancy,
            )
        }

    elif lob == LineOfBusiness.AUTO:
        col1, col2 = st.columns(2)
        with col1:
            driver_age = st.number_input("Driver Age", 16, 100, 35)
            vehicle_year = st.number_input("Vehicle Year", 1990, 2026, 2022)
            vehicle_make = st.text_input("Vehicle Make", "Toyota")
        with col2:
            annual_mileage = st.number_input("Annual Mileage", 1000, 100000, 12000)
            clean_years = st.number_input("Years Clean Record", 0, 50, 3)
            prior_accidents = st.number_input("Prior At-Fault Accidents", 0, 10, 0)
        return {
            "auto_details": AutoDetails(
                driver_age=driver_age,
                vehicle_year=vehicle_year,
                vehicle_make=vehicle_make,
                annual_mileage=annual_mileage,
                driving_record_years_clean=clean_years,
                prior_accidents=prior_accidents,
            )
        }

    elif lob == LineOfBusiness.LIFE:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Applicant Age", 18, 85, 40)
            smoker = st.checkbox("Tobacco User")
            face_amount = st.number_input(
                "Face Amount ($)", 50000, 10000000, 500000, step=50000
            )
        with col2:
            term_years = st.selectbox("Term (years)", [10, 15, 20, 25, 30], index=2)
            occupation_class = st.selectbox("Occupation Class", ["1", "2", "3", "4"])
            family_history = st.multiselect(
                "Family History Flags",
                ["cardiac", "cancer", "diabetes", "stroke"],
            )
        return {
            "life_details": LifeDetails(
                applicant_age=age,
                smoker=smoker,
                face_amount=float(face_amount),
                term_years=term_years,
                occupation_class=occupation_class,
                family_history_flags=family_history,
            )
        }

    elif lob == LineOfBusiness.COMMERCIAL:
        col1, col2 = st.columns(2)
        with col1:
            business_name = st.text_input("Business Name")
            naics_code = st.text_input("NAICS Code", "722511")
            years_in_business = st.number_input("Years in Business", 0, 100, 5)
        with col2:
            annual_revenue = st.number_input(
                "Annual Revenue ($)", 50000, 500000000, 1000000, step=100000
            )
            employee_count = st.number_input("Employee Count", 1, 10000, 25)
            prior_losses = st.number_input("Prior Losses (5 years)", 0, 50, 0)
        return {
            "commercial_details": CommercialDetails(
                business_name=business_name,
                naics_code=naics_code,
                years_in_business=years_in_business,
                annual_revenue=float(annual_revenue),
                employee_count=employee_count,
                prior_losses=prior_losses,
            )
        }

    return {}


def _display_underwriting_results():
    """Display the underwriting assessment results."""
    decision = st.session_state.underwriting_decision
    assessment = decision.risk_assessment
    coverage = decision.coverage_recommendation

    st.divider()
    st.subheader("Underwriting Assessment Results")

    # Decision banner
    decision_colors = {
        "approve": "success",
        "approve_with_conditions": "warning",
        "refer": "info",
        "decline": "error",
    }
    decision_labels = {
        "approve": "APPROVED",
        "approve_with_conditions": "APPROVED WITH CONDITIONS",
        "refer": "REFERRED",
        "decline": "DECLINED",
    }

    decision_type = decision_colors.get(decision.decision, "info")
    decision_label = decision_labels.get(decision.decision, decision.decision.upper())
    getattr(st, decision_type)(f"**Decision: {decision_label}** (Confidence: {decision.confidence_score:.0%})")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_color = (
            "normal" if assessment.overall_score < 40
            else "off" if assessment.overall_score < 65
            else "inverse"
        )
        st.metric("Risk Score", f"{assessment.overall_score:.1f}/100", delta_color=score_color)
    with col2:
        st.metric("Risk Category", assessment.risk_category.value.title())
    with col3:
        st.metric("Est. Loss Ratio", f"{(assessment.loss_ratio_estimate or 0) * 100:.1f}%")
    with col4:
        st.metric("Combined Ratio Impact", f"{decision.combined_ratio_impact or 0:.1f}%")

    # Risk Factors
    with st.expander("Risk Factor Breakdown", expanded=True):
        for factor in assessment.risk_factors:
            col_name, col_score, col_impact = st.columns([3, 1, 1])
            with col_name:
                st.write(f"**{factor.factor_name}**: {factor.description}")
            with col_score:
                st.progress(factor.score / 100.0)
                st.caption(f"{factor.score:.0f}/100 (wt: {factor.weight:.0%})")
            with col_impact:
                impact_icons = {"positive": "✅", "negative": "⚠️", "neutral": "➖"}
                st.write(impact_icons.get(factor.impact, "➖"))

    # Coverage Recommendation
    if coverage:
        with st.expander("Coverage Recommendation", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Recommended Limit:** ${coverage.recommended_limit:,.0f}")
                st.write(f"**Recommended Deductible:** ${coverage.recommended_deductible:,.0f}")
                st.write(
                    f"**Premium Range:** "
                    f"${coverage.premium_range_low:,.0f} - ${coverage.premium_range_high:,.0f}"
                )
            with col2:
                if coverage.exclusions:
                    st.write("**Exclusions:**")
                    for exc in coverage.exclusions:
                        st.write(f"- {exc}")
                if coverage.conditions:
                    st.write("**Conditions:**")
                    for cond in coverage.conditions:
                        st.write(f"- {cond}")
                if coverage.endorsements:
                    st.write("**Endorsements:**")
                    for end in coverage.endorsements:
                        st.write(f"- {end}")

    # Rationale
    with st.expander("Decision Rationale", expanded=False):
        st.write(decision.rationale)

    # Warnings
    if assessment.adverse_selection_flag:
        st.warning("**Adverse Selection Flag**: Indicators suggest potential adverse selection.")
    if assessment.moral_hazard_indicators:
        st.warning(
            "**Moral Hazard Indicators:**\n"
            + "\n".join(f"- {ind}" for ind in assessment.moral_hazard_indicators)
        )
    if decision.refer_to:
        st.info(f"**Referral Target:** {decision.refer_to}")
