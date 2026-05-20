"""Integration tests for the FastAPI application."""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.app.main import _artifacts, app
from customer_churn_ml.utils.config import load_config

client = TestClient(app)

# ---------------------------------------------------------------------------
# Minimal mock model that returns a fixed probability
# ---------------------------------------------------------------------------
class _MockModel:
    def __init__(self, proba: float = 0.85):
        self._proba = proba

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.array([[1 - self._proba, self._proba]] * n)


class _MockPreprocessor:
    def transform(self, df):
        # Return 39 features (matches current artifact shape)
        return np.zeros((len(df), 39))


@pytest.fixture
def mock_artifacts_high_churn(monkeypatch):
    """Patch _artifacts with a mock model that predicts churn (proba=0.85)."""
    from customer_churn_ml.app.main import _artifacts as artifacts

    original = dict(artifacts)
    artifacts["preprocessor"] = _MockPreprocessor()
    artifacts["model"] = _MockModel(proba=0.85)
    artifacts["threshold"] = 0.30
    artifacts["config"] = load_config()
    artifacts["shap_explainer"] = None
    artifacts["feature_names"] = None

    yield

    artifacts.clear()
    artifacts.update(original)


@pytest.fixture
def mock_artifacts_low_churn(monkeypatch):
    """Patch _artifacts with a mock model that predicts no churn (proba=0.10)."""
    from customer_churn_ml.app.main import _artifacts as artifacts

    original = dict(artifacts)
    artifacts["preprocessor"] = _MockPreprocessor()
    artifacts["model"] = _MockModel(proba=0.10)
    artifacts["threshold"] = 0.30
    artifacts["config"] = load_config()
    artifacts["shap_explainer"] = None
    artifacts["feature_names"] = None

    yield

    artifacts.clear()
    artifacts.update(original)


# ---------------------------------------------------------------------------
# Existing tests (no artifacts)
# ---------------------------------------------------------------------------
class TestAPI:
    def test_health_unhealthy_without_artifacts(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "threshold" in data
        assert "explainability_ready" in data
        assert data["explainability_ready"] is False

    def test_predict_empty_list(self):
        response = client.post("/predict", json=[])
        assert response.status_code == 422

    def test_predict_without_artifacts(self):
        payload = [{
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        }]
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "Model artifacts not loaded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests with mocked artifacts
# ---------------------------------------------------------------------------
class TestPredictWithArtifacts:
    def test_predict_without_explain(self, mock_artifacts_high_churn):
        payload = [{
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        }]
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["churn_flag"] == 1
        assert data["predictions"][0]["recommendations"] is None

    def test_predict_with_explain_churner_no_explainer(self, mock_artifacts_high_churn):
        """explain=true but no SHAP explainer loaded → churn_flag=1, recommendations=None."""
        payload = [{
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        }]
        response = client.post("/predict?explain=true", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"][0]["churn_flag"] == 1
        assert data["predictions"][0]["recommendations"] is None

    def test_predict_with_explain_non_churner(self, mock_artifacts_low_churn):
        """Non-churner should never get recommendations, even with explain=true."""
        payload = [{
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        }]
        response = client.post("/predict?explain=true", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"][0]["churn_flag"] == 0
        assert data["predictions"][0]["recommendations"] is None

    def test_explain_without_explainer(self, mock_artifacts_high_churn):
        """/explain should 503 when no SHAP explainer is loaded."""
        payload = [{
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        }]
        response = client.post("/explain", json=payload)
        assert response.status_code == 503
        assert "Explainability artifacts not loaded" in response.json()["detail"]
