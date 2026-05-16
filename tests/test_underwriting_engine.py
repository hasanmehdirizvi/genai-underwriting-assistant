"""Tests for the underwriting engine risk scoring and decision logic."""

import pytest

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
from src.underwriting_engine import UnderwritingEngine


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def engine_conservative():
    """Engine with conservative risk appetite (no Bedrock client)."""
    return UnderwritingEngine(bedrock_client=None, risk_appetite=RiskAppetite.CONSERVATIVE)


@pytest.fixture
def engine_moderate():
    """Engine with moderate risk appetite (no Bedrock client)."""
    return UnderwritingEngine(bedrock_client=None, risk_appetite=RiskAppetite.MODERATE)


@pytest.fixture
def engine_aggressive():
    """Engine with aggressive risk appetite (no Bedrock client)."""
    return UnderwritingEngine(bedrock_client=None, risk_appetite=RiskAppetite.AGGRESSIVE)


@pytest.fixture
def low_risk_property_app():
    """Low-risk property application."""
    return ApplicationData(
        applicant_name="John Smith",
        line_of_business=LineOfBusiness.PROPERTY,
        requested_coverage_limit=300000.0,
        requested_deductible=2500.0,
        property_details=PropertyDetails(
            address="123 Safe Street, Springfield, IL",
            construction_type="masonry",
            year_built=2015,
            protection_class=3,
            prior_claims=0,
            occupancy_type="owner-occupied",
        ),
    )


@pytest.fixture
def high_risk_property_app():
    """High-risk property application."""
    return ApplicationData(
        applicant_name="Jane Doe",
        line_of_business=LineOfBusiness.PROPERTY,
        requested_coverage_limit=500000.0,
        requested_deductible=1000.0,
        property_details=PropertyDetails(
            address="456 Risk Avenue, Chicago, IL",
            construction_type="frame",
            year_built=1960,
            protection_class=8,
            prior_claims=4,
            occupancy_type="vacant",
        ),
    )


@pytest.fixture
def standard_auto_app():
    """Standard auto application."""
    return ApplicationData(
        applicant_name="Bob Wilson",
        line_of_business=LineOfBusiness.AUTO,
        requested_coverage_limit=100000.0,
        requested_deductible=500.0,
        auto_details=AutoDetails(
            driver_age=35,
            vehicle_year=2022,
            vehicle_make="Honda",
            vehicle_model="Civic",
            annual_mileage=12000,
            driving_record_years_clean=5,
            prior_accidents=0,
        ),
    )


@pytest.fixture
def young_driver_auto_app():
    """High-risk young driver auto application."""
    return ApplicationData(
        applicant_name="Alex Young",
        line_of_business=LineOfBusiness.AUTO,
        requested_coverage_limit=100000.0,
        requested_deductible=500.0,
        auto_details=AutoDetails(
            driver_age=19,
            vehicle_year=2023,
            vehicle_make="BMW",
            vehicle_model="M3",
            annual_mileage=25000,
            driving_record_years_clean=0,
            prior_accidents=2,
        ),
    )


@pytest.fixture
def life_nonsmoker_app():
    """Standard life insurance application - non-smoker."""
    return ApplicationData(
        applicant_name="Sarah Johnson",
        line_of_business=LineOfBusiness.LIFE,
        requested_coverage_limit=1000000.0,
        life_details=LifeDetails(
            applicant_age=35,
            smoker=False,
            face_amount=1000000.0,
            term_years=20,
            occupation_class="1",
            family_history_flags=[],
        ),
    )


@pytest.fixture
def life_smoker_app():
    """High-risk life insurance application - smoker with history."""
    return ApplicationData(
        applicant_name="Mike Davis",
        line_of_business=LineOfBusiness.LIFE,
        requested_coverage_limit=2000000.0,
        life_details=LifeDetails(
            applicant_age=58,
            smoker=True,
            face_amount=3000000.0,
            term_years=20,
            occupation_class="3",
            family_history_flags=["cardiac", "cancer"],
        ),
    )


@pytest.fixture
def commercial_established_app():
    """Established commercial business application."""
    return ApplicationData(
        applicant_name="Acme Corp",
        line_of_business=LineOfBusiness.COMMERCIAL,
        requested_coverage_limit=2000000.0,
        requested_deductible=10000.0,
        commercial_details=CommercialDetails(
            business_name="Acme Corp",
            naics_code="541512",
            years_in_business=15,
            annual_revenue=5000000.0,
            employee_count=45,
            prior_losses=0,
            total_prior_loss_amount=0.0,
        ),
    )


# -----------------------------------------------------------------------------
# Risk Assessment Tests
# -----------------------------------------------------------------------------


class TestRiskAssessment:
    """Test risk scoring logic."""

    def test_low_risk_property_scores_low(self, engine_moderate, low_risk_property_app):
        """Low-risk property should score below 40."""
        assessment = engine_moderate.assess_risk(low_risk_property_app)
        assert assessment.overall_score < 40
        assert assessment.risk_category in (RiskCategory.PREFERRED, RiskCategory.STANDARD)

    def test_high_risk_property_scores_high(self, engine_moderate, high_risk_property_app):
        """High-risk property should score above 60."""
        assessment = engine_moderate.assess_risk(high_risk_property_app)
        assert assessment.overall_score > 60
        assert assessment.risk_category in (RiskCategory.SUBSTANDARD, RiskCategory.DECLINE)

    def test_standard_auto_scores_moderate(self, engine_moderate, standard_auto_app):
        """Standard auto should score in moderate range."""
        assessment = engine_moderate.assess_risk(standard_auto_app)
        assert 10 <= assessment.overall_score <= 50

    def test_young_driver_scores_high(self, engine_moderate, young_driver_auto_app):
        """Young driver with accidents should score high."""
        assessment = engine_moderate.assess_risk(young_driver_auto_app)
        assert assessment.overall_score > 50

    def test_nonsmoker_life_scores_low(self, engine_moderate, life_nonsmoker_app):
        """Non-smoker with no history should score low."""
        assessment = engine_moderate.assess_risk(life_nonsmoker_app)
        assert assessment.overall_score < 45

    def test_smoker_life_scores_high(self, engine_moderate, life_smoker_app):
        """Older smoker with family history should score high."""
        assessment = engine_moderate.assess_risk(life_smoker_app)
        assert assessment.overall_score > 55

    def test_risk_factors_populated(self, engine_moderate, low_risk_property_app):
        """Risk factors should be returned with valid scores and weights."""
        assessment = engine_moderate.assess_risk(low_risk_property_app)
        assert len(assessment.risk_factors) > 0
        for factor in assessment.risk_factors:
            assert 0 <= factor.score <= 100
            assert 0 <= factor.weight <= 1.0
            assert factor.factor_name != ""

    def test_loss_ratio_estimated(self, engine_moderate, standard_auto_app):
        """Loss ratio should be estimated between 0 and 1."""
        assessment = engine_moderate.assess_risk(standard_auto_app)
        assert assessment.loss_ratio_estimate is not None
        assert 0.3 <= assessment.loss_ratio_estimate <= 1.0


# -----------------------------------------------------------------------------
# Adverse Selection & Moral Hazard Tests
# -----------------------------------------------------------------------------


class TestAdverseSelectionDetection:
    """Test adverse selection and moral hazard detection."""

    def test_vacant_property_moral_hazard(self, engine_moderate, high_risk_property_app):
        """Vacant property should flag moral hazard."""
        assessment = engine_moderate.assess_risk(high_risk_property_app)
        assert len(assessment.moral_hazard_indicators) > 0
        assert any("Vacant" in ind for ind in assessment.moral_hazard_indicators)

    def test_multiple_claims_moral_hazard(self, engine_moderate, high_risk_property_app):
        """Multiple prior claims should flag moral hazard."""
        assessment = engine_moderate.assess_risk(high_risk_property_app)
        assert any("claims" in ind.lower() for ind in assessment.moral_hazard_indicators)

    def test_adverse_selection_high_limit_older_life(self, engine_moderate, life_smoker_app):
        """High face amount for older applicant should flag adverse selection."""
        assessment = engine_moderate.assess_risk(life_smoker_app)
        assert assessment.adverse_selection_flag is True

    def test_no_adverse_selection_standard_app(self, engine_moderate, standard_auto_app):
        """Standard auto app should not flag adverse selection."""
        assessment = engine_moderate.assess_risk(standard_auto_app)
        assert assessment.adverse_selection_flag is False


# -----------------------------------------------------------------------------
# Coverage Recommendation Tests
# -----------------------------------------------------------------------------


class TestCoverageRecommendation:
    """Test coverage recommendation logic."""

    def test_preferred_risk_gets_full_limit(self, engine_moderate, low_risk_property_app):
        """Preferred risk should get full requested coverage limit."""
        assessment = engine_moderate.assess_risk(low_risk_property_app)
        coverage = engine_moderate.recommend_coverage(low_risk_property_app, assessment)
        assert coverage.recommended_limit == low_risk_property_app.requested_coverage_limit

    def test_substandard_risk_gets_reduced_limit(self, engine_moderate, high_risk_property_app):
        """Substandard risk should get reduced coverage limit."""
        assessment = engine_moderate.assess_risk(high_risk_property_app)
        if assessment.risk_category == RiskCategory.SUBSTANDARD:
            coverage = engine_moderate.recommend_coverage(high_risk_property_app, assessment)
            assert coverage.recommended_limit < high_risk_property_app.requested_coverage_limit

    def test_premium_range_is_valid(self, engine_moderate, standard_auto_app):
        """Premium range should have low < high and both positive."""
        assessment = engine_moderate.assess_risk(standard_auto_app)
        coverage = engine_moderate.recommend_coverage(standard_auto_app, assessment)
        assert coverage.premium_range_low > 0
        assert coverage.premium_range_high > coverage.premium_range_low

    def test_exclusions_populated_for_higher_risk(self, engine_moderate, high_risk_property_app):
        """Higher risk applications should have exclusions applied."""
        assessment = engine_moderate.assess_risk(high_risk_property_app)
        coverage = engine_moderate.recommend_coverage(high_risk_property_app, assessment)
        assert len(coverage.exclusions) > 0


# -----------------------------------------------------------------------------
# Decision Tests
# -----------------------------------------------------------------------------


class TestUnderwritingDecision:
    """Test final underwriting decision logic."""

    def test_low_risk_approved(self, engine_moderate, low_risk_property_app):
        """Low-risk application should be approved."""
        assessment = engine_moderate.assess_risk(low_risk_property_app)
        coverage = engine_moderate.recommend_coverage(low_risk_property_app, assessment)
        decision = engine_moderate.make_decision(low_risk_property_app, assessment, coverage)
        assert decision.decision in ("approve", "approve_with_conditions")
        assert decision.confidence_score > 0.5

    def test_high_risk_not_approved_clean(self, engine_conservative, high_risk_property_app):
        """High-risk application should not get clean approval with conservative appetite."""
        assessment = engine_conservative.assess_risk(high_risk_property_app)
        coverage = engine_conservative.recommend_coverage(high_risk_property_app, assessment)
        decision = engine_conservative.make_decision(high_risk_property_app, assessment, coverage)
        assert decision.decision in ("refer", "decline")

    def test_aggressive_appetite_more_permissive(self, engine_aggressive, young_driver_auto_app):
        """Aggressive appetite should be more permissive on borderline risks."""
        assessment = engine_aggressive.assess_risk(young_driver_auto_app)
        coverage = engine_aggressive.recommend_coverage(young_driver_auto_app, assessment)
        decision = engine_aggressive.make_decision(young_driver_auto_app, assessment, coverage)
        # Aggressive should at least refer, not necessarily decline
        assert decision.decision != "decline" or assessment.overall_score > 90

    def test_decision_has_rationale(self, engine_moderate, standard_auto_app):
        """Every decision should include a rationale."""
        assessment = engine_moderate.assess_risk(standard_auto_app)
        coverage = engine_moderate.recommend_coverage(standard_auto_app, assessment)
        decision = engine_moderate.make_decision(standard_auto_app, assessment, coverage)
        assert decision.rationale != ""

    def test_referral_target_set_when_referred(self, engine_conservative, high_risk_property_app):
        """Referred decisions should specify who to refer to."""
        assessment = engine_conservative.assess_risk(high_risk_property_app)
        coverage = engine_conservative.recommend_coverage(high_risk_property_app, assessment)
        decision = engine_conservative.make_decision(high_risk_property_app, assessment, coverage)
        if decision.decision == "refer":
            assert decision.refer_to is not None
            assert "Underwriter" in decision.refer_to or "Reinsurance" in decision.refer_to

    def test_combined_ratio_impact_calculated(self, engine_moderate, commercial_established_app):
        """Combined ratio impact should be calculated."""
        assessment = engine_moderate.assess_risk(commercial_established_app)
        coverage = engine_moderate.recommend_coverage(commercial_established_app, assessment)
        decision = engine_moderate.make_decision(commercial_established_app, assessment, coverage)
        assert decision.combined_ratio_impact is not None
        assert 50 <= decision.combined_ratio_impact <= 150


# -----------------------------------------------------------------------------
# Risk Appetite Threshold Tests
# -----------------------------------------------------------------------------


class TestRiskAppetiteThresholds:
    """Test that risk appetite affects decisions correctly."""

    def test_same_app_different_outcomes(
        self,
        engine_conservative,
        engine_aggressive,
        young_driver_auto_app,
    ):
        """Same application should get different treatment under different appetites."""
        assessment_c = engine_conservative.assess_risk(young_driver_auto_app)
        coverage_c = engine_conservative.recommend_coverage(young_driver_auto_app, assessment_c)
        decision_c = engine_conservative.make_decision(
            young_driver_auto_app, assessment_c, coverage_c
        )

        assessment_a = engine_aggressive.assess_risk(young_driver_auto_app)
        coverage_a = engine_aggressive.recommend_coverage(young_driver_auto_app, assessment_a)
        decision_a = engine_aggressive.make_decision(
            young_driver_auto_app, assessment_a, coverage_a
        )

        # Conservative should be stricter
        decision_severity = {"approve": 0, "approve_with_conditions": 1, "refer": 2, "decline": 3}
        assert decision_severity[decision_c.decision] >= decision_severity[decision_a.decision]

    def test_conservative_categorizes_stricter(self, engine_conservative, engine_moderate):
        """Conservative should categorize same score as higher risk."""
        # A score of 55 should be different categories
        cat_c = engine_conservative._categorize_risk(55)
        cat_m = engine_moderate._categorize_risk(55)

        category_severity = {
            RiskCategory.PREFERRED: 0,
            RiskCategory.STANDARD: 1,
            RiskCategory.SUBSTANDARD: 2,
            RiskCategory.DECLINE: 3,
        }
        assert category_severity[cat_c] >= category_severity[cat_m]


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_property_details(self, engine_moderate):
        """Application with empty property details should still score."""
        app = ApplicationData(
            applicant_name="Empty Details",
            line_of_business=LineOfBusiness.PROPERTY,
            property_details=PropertyDetails(),
        )
        assessment = engine_moderate.assess_risk(app)
        assert 0 <= assessment.overall_score <= 100

    def test_no_details_at_all(self, engine_moderate):
        """Application with no line-specific details should handle gracefully."""
        app = ApplicationData(
            applicant_name="No Details",
            line_of_business=LineOfBusiness.PROPERTY,
            property_details=None,
        )
        assessment = engine_moderate.assess_risk(app)
        # Should return default score when no factors can be calculated
        assert assessment.overall_score == 50.0

    def test_maximum_risk_values(self, engine_moderate):
        """Application with maximum risk values should not exceed 100."""
        app = ApplicationData(
            applicant_name="Maximum Risk",
            line_of_business=LineOfBusiness.PROPERTY,
            property_details=PropertyDetails(
                construction_type="frame",
                year_built=1920,
                protection_class=10,
                prior_claims=20,
                occupancy_type="vacant",
            ),
        )
        assessment = engine_moderate.assess_risk(app)
        assert assessment.overall_score <= 100.0
