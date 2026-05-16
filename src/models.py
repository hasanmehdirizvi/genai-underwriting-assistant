"""Pydantic models for the underwriting assistant domain."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LineOfBusiness(str, Enum):
    """Insurance lines of business supported by the underwriting assistant."""

    PROPERTY = "property"
    AUTO = "auto"
    LIFE = "life"
    COMMERCIAL = "commercial"


class RiskAppetite(str, Enum):
    """Insurer's willingness to accept risk profiles."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskCategory(str, Enum):
    """Risk classification levels."""

    PREFERRED = "preferred"
    STANDARD = "standard"
    SUBSTANDARD = "substandard"
    DECLINE = "decline"


class PropertyDetails(BaseModel):
    """Property-specific application data."""

    address: str = ""
    construction_type: str = ""  # frame, masonry, fire-resistive
    year_built: Optional[int] = None
    square_footage: Optional[int] = None
    roof_type: str = ""
    protection_class: Optional[int] = Field(None, ge=1, le=10)
    prior_claims: int = 0
    occupancy_type: str = ""  # owner-occupied, tenant, vacant


class AutoDetails(BaseModel):
    """Auto-specific application data."""

    vehicle_year: Optional[int] = None
    vehicle_make: str = ""
    vehicle_model: str = ""
    vin: str = ""
    annual_mileage: Optional[int] = None
    driver_age: Optional[int] = None
    driving_record_years_clean: int = 0
    prior_accidents: int = 0
    prior_violations: int = 0
    garage_zip: str = ""


class LifeDetails(BaseModel):
    """Life insurance-specific application data."""

    applicant_age: Optional[int] = None
    gender: str = ""
    smoker: bool = False
    face_amount: Optional[float] = None
    term_years: Optional[int] = None
    occupation_class: str = ""  # 1-4, 1 being lowest risk
    family_history_flags: list[str] = Field(default_factory=list)


class CommercialDetails(BaseModel):
    """Commercial lines-specific application data."""

    business_name: str = ""
    naics_code: str = ""
    years_in_business: Optional[int] = None
    annual_revenue: Optional[float] = None
    employee_count: Optional[int] = None
    prior_losses: int = 0
    total_prior_loss_amount: float = 0.0
    operations_description: str = ""


class ApplicationData(BaseModel):
    """Unified insurance application data model."""

    application_id: str = ""
    line_of_business: LineOfBusiness = LineOfBusiness.PROPERTY
    applicant_name: str = ""
    application_date: date = Field(default_factory=date.today)
    effective_date: Optional[date] = None
    requested_coverage_limit: Optional[float] = None
    requested_deductible: Optional[float] = None

    # Line-specific details (only one populated based on LOB)
    property_details: Optional[PropertyDetails] = None
    auto_details: Optional[AutoDetails] = None
    life_details: Optional[LifeDetails] = None
    commercial_details: Optional[CommercialDetails] = None

    # Additional context
    notes: str = ""


class RiskFactor(BaseModel):
    """Individual risk factor contributing to overall assessment."""

    factor_name: str
    description: str
    score: float = Field(ge=0.0, le=100.0)
    weight: float = Field(ge=0.0, le=1.0)
    impact: str = ""  # positive, negative, neutral


class RiskAssessment(BaseModel):
    """Complete risk assessment output."""

    overall_score: float = Field(ge=0.0, le=100.0)
    risk_category: RiskCategory
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    loss_ratio_estimate: Optional[float] = None
    adverse_selection_flag: bool = False
    moral_hazard_indicators: list[str] = Field(default_factory=list)
    explanation: str = ""


class CoverageRecommendation(BaseModel):
    """Recommended coverage structure."""

    recommended_limit: float
    recommended_deductible: float
    premium_range_low: float
    premium_range_high: float
    exclusions: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    endorsements: list[str] = Field(default_factory=list)


class UnderwritingDecision(BaseModel):
    """Final underwriting decision with full context."""

    decision: str  # approve, approve_with_conditions, refer, decline
    risk_assessment: RiskAssessment
    coverage_recommendation: Optional[CoverageRecommendation] = None
    rationale: str = ""
    refer_to: Optional[str] = None  # senior underwriter, reinsurance, etc.
    combined_ratio_impact: Optional[float] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    guardrail_interventions: list[str] = Field(default_factory=list)
