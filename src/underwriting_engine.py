"""Core underwriting logic for risk scoring, coverage recommendations, and decisions.

Implements rule-based risk assessment combined with LLM-powered analysis
for explainable underwriting decisions across multiple lines of business.
"""

import json
import logging
from typing import Optional

from src.bedrock_client import BedrockClient, GuardrailIntervention
from src.models import (
    ApplicationData,
    CoverageRecommendation,
    LineOfBusiness,
    RiskAppetite,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    UnderwritingDecision,
)
from src.prompts import (
    COVERAGE_RECOMMENDATION_PROMPT,
    DECISION_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# Risk scoring weights by line of business
RISK_WEIGHTS = {
    LineOfBusiness.PROPERTY: {
        "protection_class": 0.20,
        "construction": 0.15,
        "age_of_structure": 0.15,
        "prior_claims": 0.25,
        "occupancy": 0.10,
        "location": 0.15,
    },
    LineOfBusiness.AUTO: {
        "driver_age": 0.15,
        "driving_record": 0.30,
        "vehicle_age": 0.10,
        "annual_mileage": 0.10,
        "prior_accidents": 0.25,
        "territory": 0.10,
    },
    LineOfBusiness.LIFE: {
        "applicant_age": 0.25,
        "smoker_status": 0.25,
        "face_amount": 0.15,
        "occupation": 0.15,
        "family_history": 0.20,
    },
    LineOfBusiness.COMMERCIAL: {
        "industry_class": 0.20,
        "years_in_business": 0.15,
        "loss_history": 0.25,
        "revenue_size": 0.10,
        "employee_count": 0.10,
        "operations_complexity": 0.20,
    },
}

# Risk appetite thresholds
APPETITE_THRESHOLDS = {
    RiskAppetite.CONSERVATIVE: {
        "max_score_for_approve": 40,
        "max_score_for_conditions": 55,
        "max_score_for_refer": 70,
        # Above 70 = decline
    },
    RiskAppetite.MODERATE: {
        "max_score_for_approve": 50,
        "max_score_for_conditions": 65,
        "max_score_for_refer": 80,
    },
    RiskAppetite.AGGRESSIVE: {
        "max_score_for_approve": 60,
        "max_score_for_conditions": 75,
        "max_score_for_refer": 90,
    },
}


class UnderwritingEngine:
    """Core underwriting engine combining rule-based and LLM-powered assessment.

    Provides:
    - Deterministic risk scoring based on underwriting factors
    - LLM-enhanced risk analysis for nuanced assessment
    - Coverage recommendations aligned with risk appetite
    - Final decision logic with referral routing
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        risk_appetite: RiskAppetite = RiskAppetite.MODERATE,
    ):
        self.bedrock_client = bedrock_client
        self.risk_appetite = risk_appetite

    def assess_risk(self, application: ApplicationData) -> RiskAssessment:
        """Perform risk assessment on an insurance application.

        Combines deterministic scoring with LLM analysis for explainability.

        Args:
            application: Complete application data.

        Returns:
            RiskAssessment with scores, factors, and explanation.
        """
        # Step 1: Deterministic risk factor scoring
        risk_factors = self._score_risk_factors(application)

        # Step 2: Calculate overall score
        overall_score = self._calculate_overall_score(risk_factors)

        # Step 3: Determine risk category
        risk_category = self._categorize_risk(overall_score)

        # Step 4: Check for adverse selection and moral hazard
        adverse_selection = self._check_adverse_selection(application)
        moral_hazard = self._check_moral_hazard(application)

        # Step 5: Estimate loss ratio
        loss_ratio = self._estimate_loss_ratio(overall_score, application.line_of_business)

        # Step 6: Generate explanation via LLM (if client available)
        explanation = self._generate_explanation(application, risk_factors, overall_score)

        return RiskAssessment(
            overall_score=overall_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            loss_ratio_estimate=loss_ratio,
            adverse_selection_flag=adverse_selection,
            moral_hazard_indicators=moral_hazard,
            explanation=explanation,
        )

    def recommend_coverage(
        self,
        application: ApplicationData,
        risk_assessment: RiskAssessment,
    ) -> CoverageRecommendation:
        """Generate coverage recommendation based on risk assessment.

        Args:
            application: Original application data.
            risk_assessment: Completed risk assessment.

        Returns:
            CoverageRecommendation with limits, deductible, premium range.
        """
        # Base coverage on requested amounts adjusted for risk
        requested_limit = application.requested_coverage_limit or self._default_limit(
            application.line_of_business
        )
        requested_deductible = application.requested_deductible or self._default_deductible(
            application.line_of_business
        )

        # Adjust based on risk category
        limit_multiplier = self._limit_multiplier(risk_assessment.risk_category)
        deductible_multiplier = self._deductible_multiplier(risk_assessment.risk_category)

        recommended_limit = requested_limit * limit_multiplier
        recommended_deductible = requested_deductible * deductible_multiplier

        # Calculate premium range
        base_rate = self._base_rate(application.line_of_business, recommended_limit)
        risk_load = 1.0 + (risk_assessment.overall_score / 100.0) * 0.5
        premium_low = base_rate * risk_load * 0.9
        premium_high = base_rate * risk_load * 1.1

        # Determine exclusions and conditions
        exclusions = self._determine_exclusions(application, risk_assessment)
        conditions = self._determine_conditions(application, risk_assessment)
        endorsements = self._determine_endorsements(application, risk_assessment)

        return CoverageRecommendation(
            recommended_limit=recommended_limit,
            recommended_deductible=recommended_deductible,
            premium_range_low=round(premium_low, 2),
            premium_range_high=round(premium_high, 2),
            exclusions=exclusions,
            conditions=conditions,
            endorsements=endorsements,
        )

    def make_decision(
        self,
        application: ApplicationData,
        risk_assessment: RiskAssessment,
        coverage_recommendation: CoverageRecommendation,
    ) -> UnderwritingDecision:
        """Make final underwriting decision.

        Applies risk appetite thresholds to determine approve/refer/decline.

        Args:
            application: Original application.
            risk_assessment: Risk assessment results.
            coverage_recommendation: Recommended coverage structure.

        Returns:
            UnderwritingDecision with decision, rationale, and confidence.
        """
        thresholds = APPETITE_THRESHOLDS[self.risk_appetite]
        score = risk_assessment.overall_score
        guardrail_interventions: list[str] = []

        # Decision logic based on risk appetite thresholds
        if score <= thresholds["max_score_for_approve"]:
            decision = "approve"
            refer_to = None
            confidence = 0.9 - (score / 100.0) * 0.3
        elif score <= thresholds["max_score_for_conditions"]:
            decision = "approve_with_conditions"
            refer_to = None
            confidence = 0.75 - (score / 100.0) * 0.2
        elif score <= thresholds["max_score_for_refer"]:
            decision = "refer"
            refer_to = self._determine_referral(application, risk_assessment)
            confidence = 0.6
        else:
            decision = "decline"
            refer_to = None
            confidence = 0.85

        # Override: always refer if adverse selection flagged
        if risk_assessment.adverse_selection_flag and decision == "approve":
            decision = "approve_with_conditions"
            confidence *= 0.8

        # Generate rationale via LLM
        rationale = self._generate_rationale(
            application, risk_assessment, coverage_recommendation, decision
        )

        # Estimate combined ratio impact
        combined_ratio_impact = self._estimate_combined_ratio_impact(
            risk_assessment, coverage_recommendation
        )

        return UnderwritingDecision(
            decision=decision,
            risk_assessment=risk_assessment,
            coverage_recommendation=coverage_recommendation,
            rationale=rationale,
            refer_to=refer_to,
            combined_ratio_impact=combined_ratio_impact,
            confidence_score=round(confidence, 2),
            guardrail_interventions=guardrail_interventions,
        )

    # -------------------------------------------------------------------------
    # Private methods: Risk Factor Scoring
    # -------------------------------------------------------------------------

    def _score_risk_factors(self, application: ApplicationData) -> list[RiskFactor]:
        """Score individual risk factors based on line of business."""
        factors = []

        if application.line_of_business == LineOfBusiness.PROPERTY:
            factors = self._score_property_factors(application)
        elif application.line_of_business == LineOfBusiness.AUTO:
            factors = self._score_auto_factors(application)
        elif application.line_of_business == LineOfBusiness.LIFE:
            factors = self._score_life_factors(application)
        elif application.line_of_business == LineOfBusiness.COMMERCIAL:
            factors = self._score_commercial_factors(application)

        return factors

    def _score_property_factors(self, application: ApplicationData) -> list[RiskFactor]:
        """Score property-specific risk factors."""
        details = application.property_details
        if not details:
            return []

        weights = RISK_WEIGHTS[LineOfBusiness.PROPERTY]
        factors = []

        # Protection class (1=best, 10=worst)
        pc = details.protection_class or 5
        pc_score = (pc / 10.0) * 100
        factors.append(RiskFactor(
            factor_name="Protection Class",
            description=f"ISO protection class {pc} (fire department proximity/capability)",
            score=pc_score,
            weight=weights["protection_class"],
            impact="negative" if pc > 5 else "positive",
        ))

        # Construction type
        construction_scores = {
            "fire-resistive": 15,
            "masonry": 30,
            "frame": 55,
            "": 50,
        }
        cs = construction_scores.get(details.construction_type.lower(), 50)
        factors.append(RiskFactor(
            factor_name="Construction Type",
            description=f"{details.construction_type or 'Unknown'} construction",
            score=cs,
            weight=weights["construction"],
            impact="negative" if cs > 40 else "positive",
        ))

        # Age of structure
        import datetime
        current_year = datetime.datetime.now().year
        age = current_year - (details.year_built or current_year - 20)
        age_score = min(100, (age / 50.0) * 100)
        factors.append(RiskFactor(
            factor_name="Structure Age",
            description=f"Built {details.year_built or 'unknown'} ({age} years old)",
            score=age_score,
            weight=weights["age_of_structure"],
            impact="negative" if age > 30 else "neutral",
        ))

        # Prior claims
        claims_score = min(100, details.prior_claims * 30)
        factors.append(RiskFactor(
            factor_name="Prior Claims",
            description=f"{details.prior_claims} prior claims on record",
            score=claims_score,
            weight=weights["prior_claims"],
            impact="negative" if details.prior_claims > 0 else "positive",
        ))

        # Occupancy
        occupancy_scores = {
            "owner-occupied": 20,
            "tenant": 40,
            "vacant": 80,
            "": 50,
        }
        occ_score = occupancy_scores.get(details.occupancy_type.lower(), 50)
        factors.append(RiskFactor(
            factor_name="Occupancy Type",
            description=f"{details.occupancy_type or 'Unknown'} occupancy",
            score=occ_score,
            weight=weights["occupancy"],
            impact="negative" if occ_score > 40 else "positive",
        ))

        return factors

    def _score_auto_factors(self, application: ApplicationData) -> list[RiskFactor]:
        """Score auto-specific risk factors."""
        details = application.auto_details
        if not details:
            return []

        weights = RISK_WEIGHTS[LineOfBusiness.AUTO]
        factors = []

        # Driver age
        age = details.driver_age or 35
        if age < 25:
            age_score = 70
        elif age < 30:
            age_score = 45
        elif age <= 65:
            age_score = 20
        else:
            age_score = 50
        factors.append(RiskFactor(
            factor_name="Driver Age",
            description=f"Driver age {age}",
            score=age_score,
            weight=weights["driver_age"],
            impact="negative" if age_score > 40 else "positive",
        ))

        # Driving record
        clean_years = details.driving_record_years_clean
        record_score = max(0, 80 - clean_years * 15)
        factors.append(RiskFactor(
            factor_name="Driving Record",
            description=f"{clean_years} years clean driving record",
            score=record_score,
            weight=weights["driving_record"],
            impact="positive" if clean_years >= 3 else "negative",
        ))

        # Prior accidents
        accident_score = min(100, details.prior_accidents * 35)
        factors.append(RiskFactor(
            factor_name="Prior Accidents",
            description=f"{details.prior_accidents} prior at-fault accidents",
            score=accident_score,
            weight=weights["prior_accidents"],
            impact="negative" if details.prior_accidents > 0 else "positive",
        ))

        # Annual mileage
        mileage = details.annual_mileage or 12000
        if mileage < 7500:
            mileage_score = 15
        elif mileage < 12000:
            mileage_score = 30
        elif mileage < 20000:
            mileage_score = 50
        else:
            mileage_score = 70
        factors.append(RiskFactor(
            factor_name="Annual Mileage",
            description=f"{mileage:,} miles per year",
            score=mileage_score,
            weight=weights["annual_mileage"],
            impact="negative" if mileage > 15000 else "neutral",
        ))

        return factors

    def _score_life_factors(self, application: ApplicationData) -> list[RiskFactor]:
        """Score life insurance-specific risk factors."""
        details = application.life_details
        if not details:
            return []

        weights = RISK_WEIGHTS[LineOfBusiness.LIFE]
        factors = []

        # Applicant age
        age = details.applicant_age or 40
        if age < 30:
            age_score = 15
        elif age < 45:
            age_score = 30
        elif age < 55:
            age_score = 50
        elif age < 65:
            age_score = 70
        else:
            age_score = 85
        factors.append(RiskFactor(
            factor_name="Applicant Age",
            description=f"Age {age} at application",
            score=age_score,
            weight=weights["applicant_age"],
            impact="negative" if age > 50 else "neutral",
        ))

        # Smoker status
        smoker_score = 75 if details.smoker else 10
        factors.append(RiskFactor(
            factor_name="Tobacco Use",
            description="Tobacco user" if details.smoker else "Non-tobacco",
            score=smoker_score,
            weight=weights["smoker_status"],
            impact="negative" if details.smoker else "positive",
        ))

        # Face amount relative to typical
        face = details.face_amount or 500000
        if face < 250000:
            face_score = 20
        elif face < 1000000:
            face_score = 35
        elif face < 5000000:
            face_score = 55
        else:
            face_score = 75
        factors.append(RiskFactor(
            factor_name="Face Amount",
            description=f"${face:,.0f} requested coverage",
            score=face_score,
            weight=weights["face_amount"],
            impact="neutral",
        ))

        # Family history
        history_count = len(details.family_history_flags)
        history_score = min(100, history_count * 25)
        factors.append(RiskFactor(
            factor_name="Family History",
            description=f"{history_count} family history flag(s) noted",
            score=history_score,
            weight=weights["family_history"],
            impact="negative" if history_count > 0 else "positive",
        ))

        return factors

    def _score_commercial_factors(self, application: ApplicationData) -> list[RiskFactor]:
        """Score commercial lines-specific risk factors."""
        details = application.commercial_details
        if not details:
            return []

        weights = RISK_WEIGHTS[LineOfBusiness.COMMERCIAL]
        factors = []

        # Years in business
        years = details.years_in_business or 0
        if years < 2:
            years_score = 70
        elif years < 5:
            years_score = 45
        elif years < 10:
            years_score = 25
        else:
            years_score = 15
        factors.append(RiskFactor(
            factor_name="Business Tenure",
            description=f"{years} years in operation",
            score=years_score,
            weight=weights["years_in_business"],
            impact="positive" if years >= 5 else "negative",
        ))

        # Loss history
        losses = details.prior_losses
        loss_score = min(100, losses * 25)
        factors.append(RiskFactor(
            factor_name="Loss History",
            description=f"{losses} prior losses totaling ${details.total_prior_loss_amount:,.0f}",
            score=loss_score,
            weight=weights["loss_history"],
            impact="negative" if losses > 0 else "positive",
        ))

        # Employee count (complexity proxy)
        employees = details.employee_count or 10
        if employees < 10:
            emp_score = 20
        elif employees < 50:
            emp_score = 35
        elif employees < 200:
            emp_score = 50
        else:
            emp_score = 65
        factors.append(RiskFactor(
            factor_name="Employee Count",
            description=f"{employees} employees (operations complexity indicator)",
            score=emp_score,
            weight=weights["employee_count"],
            impact="neutral",
        ))

        return factors

    # -------------------------------------------------------------------------
    # Private methods: Calculations
    # -------------------------------------------------------------------------

    def _calculate_overall_score(self, factors: list[RiskFactor]) -> float:
        """Calculate weighted average risk score."""
        if not factors:
            return 50.0

        total_weighted = sum(f.score * f.weight for f in factors)
        total_weight = sum(f.weight for f in factors)

        if total_weight == 0:
            return 50.0

        return round(total_weighted / total_weight, 1)

    def _categorize_risk(self, score: float) -> RiskCategory:
        """Map overall score to risk category."""
        thresholds = APPETITE_THRESHOLDS[self.risk_appetite]

        if score <= thresholds["max_score_for_approve"]:
            return RiskCategory.PREFERRED
        elif score <= thresholds["max_score_for_conditions"]:
            return RiskCategory.STANDARD
        elif score <= thresholds["max_score_for_refer"]:
            return RiskCategory.SUBSTANDARD
        else:
            return RiskCategory.DECLINE

    def _check_adverse_selection(self, application: ApplicationData) -> bool:
        """Detect potential adverse selection indicators."""
        # High limits with minimal history
        if application.line_of_business == LineOfBusiness.LIFE:
            details = application.life_details
            if details and details.face_amount and details.face_amount > 2000000:
                if details.applicant_age and details.applicant_age > 55:
                    return True

        # Property with recent purchase and high limits
        if application.line_of_business == LineOfBusiness.PROPERTY:
            details = application.property_details
            if details and details.prior_claims >= 3:
                return True

        return False

    def _check_moral_hazard(self, application: ApplicationData) -> list[str]:
        """Identify moral hazard indicators."""
        indicators = []

        if application.line_of_business == LineOfBusiness.PROPERTY:
            details = application.property_details
            if details:
                if details.occupancy_type.lower() == "vacant":
                    indicators.append("Vacant property - increased moral hazard")
                if details.prior_claims >= 2:
                    indicators.append("Multiple prior claims - potential pattern")

        if application.line_of_business == LineOfBusiness.COMMERCIAL:
            details = application.commercial_details
            if details:
                if details.years_in_business and details.years_in_business < 1:
                    indicators.append("New business with limited track record")
                if details.total_prior_loss_amount > 100000:
                    indicators.append("Significant prior loss amount")

        return indicators

    def _estimate_loss_ratio(self, score: float, lob: LineOfBusiness) -> float:
        """Estimate expected loss ratio based on risk score."""
        # Base loss ratios by LOB
        base_ratios = {
            LineOfBusiness.PROPERTY: 0.55,
            LineOfBusiness.AUTO: 0.65,
            LineOfBusiness.LIFE: 0.45,
            LineOfBusiness.COMMERCIAL: 0.60,
        }
        base = base_ratios.get(lob, 0.60)

        # Adjust for risk score (higher score = higher expected losses)
        adjustment = (score / 100.0) * 0.3
        return round(base + adjustment, 3)

    # -------------------------------------------------------------------------
    # Private methods: Coverage
    # -------------------------------------------------------------------------

    def _default_limit(self, lob: LineOfBusiness) -> float:
        """Default coverage limit by line of business."""
        defaults = {
            LineOfBusiness.PROPERTY: 300000.0,
            LineOfBusiness.AUTO: 100000.0,
            LineOfBusiness.LIFE: 500000.0,
            LineOfBusiness.COMMERCIAL: 1000000.0,
        }
        return defaults.get(lob, 500000.0)

    def _default_deductible(self, lob: LineOfBusiness) -> float:
        """Default deductible by line of business."""
        defaults = {
            LineOfBusiness.PROPERTY: 2500.0,
            LineOfBusiness.AUTO: 500.0,
            LineOfBusiness.LIFE: 0.0,
            LineOfBusiness.COMMERCIAL: 5000.0,
        }
        return defaults.get(lob, 1000.0)

    def _limit_multiplier(self, category: RiskCategory) -> float:
        """Coverage limit adjustment based on risk category."""
        multipliers = {
            RiskCategory.PREFERRED: 1.0,
            RiskCategory.STANDARD: 1.0,
            RiskCategory.SUBSTANDARD: 0.75,
            RiskCategory.DECLINE: 0.0,
        }
        return multipliers.get(category, 1.0)

    def _deductible_multiplier(self, category: RiskCategory) -> float:
        """Deductible adjustment based on risk category (higher risk = higher deductible)."""
        multipliers = {
            RiskCategory.PREFERRED: 1.0,
            RiskCategory.STANDARD: 1.0,
            RiskCategory.SUBSTANDARD: 2.0,
            RiskCategory.DECLINE: 1.0,
        }
        return multipliers.get(category, 1.0)

    def _base_rate(self, lob: LineOfBusiness, limit: float) -> float:
        """Calculate base rate (premium per $1000 of coverage)."""
        rates_per_thousand = {
            LineOfBusiness.PROPERTY: 3.50,
            LineOfBusiness.AUTO: 8.00,
            LineOfBusiness.LIFE: 1.20,
            LineOfBusiness.COMMERCIAL: 5.00,
        }
        rate = rates_per_thousand.get(lob, 5.00)
        return (limit / 1000.0) * rate

    def _determine_exclusions(
        self,
        application: ApplicationData,
        assessment: RiskAssessment,
    ) -> list[str]:
        """Determine appropriate policy exclusions."""
        exclusions = []

        if application.line_of_business == LineOfBusiness.PROPERTY:
            if assessment.overall_score > 60:
                exclusions.append("Earth movement and mine subsidence")
            exclusions.append("Flood (separate policy required)")
            exclusions.append("Ordinance or law beyond 10% of Coverage A")

        elif application.line_of_business == LineOfBusiness.AUTO:
            if assessment.overall_score > 50:
                exclusions.append("Racing or speed contests")
                exclusions.append("Ride-sharing commercial use")

        elif application.line_of_business == LineOfBusiness.COMMERCIAL:
            exclusions.append("Pollution and environmental liability")
            exclusions.append("Professional liability (separate policy)")
            if assessment.overall_score > 55:
                exclusions.append("Product recall expenses")

        return exclusions

    def _determine_conditions(
        self,
        application: ApplicationData,
        assessment: RiskAssessment,
    ) -> list[str]:
        """Determine policy conditions based on risk assessment."""
        conditions = []

        if assessment.overall_score > 50:
            conditions.append("Annual inspection required for continued coverage")

        if assessment.adverse_selection_flag:
            conditions.append("Waiting period: 30-day elimination period on claims")

        if assessment.moral_hazard_indicators:
            conditions.append("Quarterly risk management reporting required")

        if application.line_of_business == LineOfBusiness.PROPERTY:
            details = application.property_details
            if details and details.year_built and details.year_built < 1980:
                conditions.append("Electrical and plumbing inspection within 60 days")

        return conditions

    def _determine_endorsements(
        self,
        application: ApplicationData,
        assessment: RiskAssessment,
    ) -> list[str]:
        """Recommend policy endorsements."""
        endorsements = []

        if application.line_of_business == LineOfBusiness.PROPERTY:
            endorsements.append("Replacement cost endorsement")
            if assessment.overall_score < 40:
                endorsements.append("Equipment breakdown coverage")

        elif application.line_of_business == LineOfBusiness.COMMERCIAL:
            endorsements.append("Business income with extra expense")
            endorsements.append("Employee dishonesty")

        return endorsements

    def _determine_referral(
        self,
        application: ApplicationData,
        assessment: RiskAssessment,
    ) -> str:
        """Determine who should review referred risks."""
        if assessment.overall_score > 80:
            return "Senior Underwriter - high severity risk"

        if application.requested_coverage_limit and application.requested_coverage_limit > 5000000:
            return "Facultative Reinsurance - large limit exposure"

        if assessment.adverse_selection_flag:
            return "Senior Underwriter - adverse selection indicators"

        return "Senior Underwriter - standard referral"

    def _estimate_combined_ratio_impact(
        self,
        assessment: RiskAssessment,
        coverage: CoverageRecommendation,
    ) -> float:
        """Estimate impact on book combined ratio."""
        # Simplified: higher risk = more negative impact on combined ratio
        loss_ratio = assessment.loss_ratio_estimate or 0.6
        expense_ratio = 0.30  # Industry average expense ratio
        combined = loss_ratio + expense_ratio
        return round(combined * 100, 1)

    # -------------------------------------------------------------------------
    # Private methods: LLM Integration
    # -------------------------------------------------------------------------

    def _generate_explanation(
        self,
        application: ApplicationData,
        factors: list[RiskFactor],
        score: float,
    ) -> str:
        """Generate human-readable risk explanation using LLM."""
        if not self.bedrock_client:
            # Fallback without LLM
            factor_summary = ", ".join(
                f"{f.factor_name} ({f.score:.0f})" for f in factors[:3]
            )
            return (
                f"Risk score of {score:.1f}/100 based on primary factors: {factor_summary}. "
                f"Risk appetite is {self.risk_appetite.value}."
            )

        prompt = RISK_ASSESSMENT_PROMPT.format(
            application_data=application.model_dump_json(indent=2),
            line_of_business=application.line_of_business.value,
            risk_appetite=self.risk_appetite.value,
        )

        try:
            response = self.bedrock_client.converse(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
            )
            return response["content"]
        except GuardrailIntervention as e:
            logger.warning("Guardrail intervention during explanation: %s", str(e))
            return f"Risk score: {score:.1f}/100. [Detailed explanation filtered by safety controls]"
        except Exception as e:
            logger.error("LLM explanation generation failed: %s", str(e))
            return f"Risk score: {score:.1f}/100. Unable to generate detailed explanation."

    def _generate_rationale(
        self,
        application: ApplicationData,
        assessment: RiskAssessment,
        coverage: CoverageRecommendation,
        decision: str,
    ) -> str:
        """Generate decision rationale using LLM."""
        if not self.bedrock_client:
            return (
                f"Decision: {decision}. Overall risk score {assessment.overall_score:.1f}/100 "
                f"({assessment.risk_category.value}). "
                f"Risk appetite: {self.risk_appetite.value}."
            )

        prompt = DECISION_PROMPT.format(
            application_summary=json.dumps({
                "line_of_business": application.line_of_business.value,
                "applicant": application.applicant_name,
                "requested_limit": application.requested_coverage_limit,
            }),
            risk_assessment=json.dumps({
                "score": assessment.overall_score,
                "category": assessment.risk_category.value,
                "loss_ratio": assessment.loss_ratio_estimate,
                "adverse_selection": assessment.adverse_selection_flag,
            }),
            coverage_recommendation=json.dumps({
                "limit": coverage.recommended_limit,
                "deductible": coverage.recommended_deductible,
                "premium_range": f"${coverage.premium_range_low:,.0f} - ${coverage.premium_range_high:,.0f}",
            }),
            risk_appetite=self.risk_appetite.value,
        )

        try:
            response = self.bedrock_client.converse(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
            )
            return response["content"]
        except GuardrailIntervention as e:
            logger.warning("Guardrail intervention during rationale: %s", str(e))
            return f"Decision: {decision}. [Detailed rationale filtered by safety controls]"
        except Exception as e:
            logger.error("LLM rationale generation failed: %s", str(e))
            return f"Decision: {decision}. Score: {assessment.overall_score:.1f}/100."
