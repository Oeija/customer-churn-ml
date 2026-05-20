"""Integration tests for the FastAPI application."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.app.main import app

client = TestClient(app)


class TestAPI:
    def test_health_unhealthy_without_artifacts(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "threshold" in data

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
