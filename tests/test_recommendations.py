"""Unit tests for the recommendation engine."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.app.recommendations import (
    FEATURE_DISPLAY_NAMES,
    _get_display_name,
    build_recommendations,
    generate_recommendation,
)


class TestDisplayNames:
    def test_known_feature(self):
        assert _get_display_name("contract_Two year") == "Two-Year Contract"

    def test_unknown_feature_fallback(self):
        assert _get_display_name("unknown_feature_xyz") == "unknown_feature_xyz"

    def test_all_features_have_display_name(self):
        """Every key in FEATURE_DISPLAY_NAMES should have a non-empty value."""
        for k, v in FEATURE_DISPLAY_NAMES.items():
            assert v and isinstance(v, str)


class TestGenerateRecommendation:
    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "tenure": 24,
            "contract": "Month-to-month",
            "internet_service": "DSL",
            "payment_method": "Electronic check",
            "monthly_charges": 70.0,
            "total_charges": 500.0,
            "online_security": "No",
            "tech_support": "No",
            "num_services": 1,
            "paperless_billing": "Yes",
            "senior_citizen": 0,
            "multiple_lines": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "online_backup": "No",
            "device_protection": "No",
            "has_internet": 1,
            "has_phone": 1,
            "tenure_group": "25-48",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_contract_two_year_missing(self):
        row = self._make_row(contract="Month-to-month")
        rec = generate_recommendation("contract_Two year", row)
        assert rec is not None
        assert "two-year contract" in rec.lower()

    def test_contract_two_year_present(self):
        row = self._make_row(contract="Two year")
        rec = generate_recommendation("contract_Two year", row)
        assert rec is None  # No recommendation needed — already has it

    def test_tenure_short(self):
        row = self._make_row(tenure=3)
        rec = generate_recommendation("tenure", row)
        assert rec is not None
        assert "first 3–6 months" in rec

    def test_tenure_medium(self):
        row = self._make_row(tenure=8)
        rec = generate_recommendation("tenure", row)
        assert rec is not None
        assert "first year" in rec

    def test_tenure_long(self):
        row = self._make_row(tenure=36)
        rec = generate_recommendation("tenure", row)
        assert rec is not None
        assert "loyalty rewards" in rec

    def test_internet_fiber_optic(self):
        row = self._make_row(internet_service="Fiber optic")
        rec = generate_recommendation("internet_service_Fiber optic", row)
        assert rec is not None
        assert "fibre optic" in rec.lower()

    def test_payment_electronic_check(self):
        row = self._make_row(payment_method="Electronic check")
        rec = generate_recommendation("payment_method_Electronic check", row)
        assert rec is not None
        assert "automatic payment" in rec.lower()

    def test_monthly_charges_high(self):
        row = self._make_row(monthly_charges=90.0)
        rec = generate_recommendation("monthly_charges", row)
        assert rec is not None
        assert "pricing plan" in rec.lower()

    def test_monthly_charges_low(self):
        row = self._make_row(monthly_charges=30.0)
        rec = generate_recommendation("monthly_charges", row)
        assert rec is None

    def test_online_security_missing(self):
        row = self._make_row(online_security="No")
        rec = generate_recommendation("online_security_Yes", row)
        assert rec is not None
        assert "online security" in rec.lower()

    def test_online_security_present(self):
        row = self._make_row(online_security="Yes")
        rec = generate_recommendation("online_security_Yes", row)
        assert rec is None

    def test_no_match_returns_none(self):
        row = self._make_row()
        rec = generate_recommendation("contract_One year", row)
        # contract is Month-to-month, so One year recommendation applies
        assert rec is not None


class TestBuildRecommendations:
    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "tenure": 5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "monthly_charges": 85.0,
            "total_charges": 50.0,
            "online_security": "No",
            "tech_support": "No",
            "num_services": 1,
            "paperless_billing": "Yes",
            "senior_citizen": 0,
            "multiple_lines": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "online_backup": "No",
            "device_protection": "No",
            "has_internet": 1,
            "has_phone": 1,
            "tenure_group": "0-12",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_build_recommendations_structure(self):
        top_features = [
            ("tenure", 0.45, 5),
            ("contract_Two year", 0.30, 0),
            ("internet_service_Fiber optic", 0.25, 1),
        ]
        row = self._make_row()
        recs = build_recommendations(top_features, row)

        assert len(recs) == 3
        for rec in recs:
            assert "feature" in rec
            assert "display_name" in rec
            assert "shap_value" in rec
            assert "feature_value" in rec
            assert "recommendation" in rec
            assert isinstance(rec["recommendation"], str)
            assert len(rec["recommendation"]) > 0

    def test_display_names_populated(self):
        top_features = [
            ("contract_Two year", 0.30, 0),
        ]
        row = self._make_row()
        recs = build_recommendations(top_features, row)
        assert recs[0]["display_name"] == "Two-Year Contract"
