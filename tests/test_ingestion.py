"""Unit tests for data ingestion."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.data.ingestion import load_raw_data, standardise_columns


@pytest.fixture
def mock_raw_csv(tmp_path):
    """Create a minimal mock CSV mimicking the Telco dataset."""
    df = pd.DataFrame({
        "customerID": ["A1", "A2", "A3"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 34, 2],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No"],
        "InternetService": ["DSL", "DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", "1889.5", ""],  # blank string for tenure=0 case
        "Churn": ["No", "No", "Yes"],
    })
    path = tmp_path / "mock_telco.csv"
    df.to_csv(path, index=False)
    return path


class TestIngestion:
    def test_standardise_columns(self):
        df = pd.DataFrame({"CustomerID": [1], "MonthlyCharges": [50]})
        result = standardise_columns(df)
        assert list(result.columns) == ["customer_id", "monthly_charges"]

    def test_load_raw_data(self, mock_raw_csv):
        df = load_raw_data(str(mock_raw_csv))
        assert df.shape == (3, 20)  # 21 cols - customer_id dropped
        assert "customer_id" not in df.columns
        assert "total_charges" in df.columns
        assert df["total_charges"].isna().sum() == 1  # blank string → NaN
        assert df["churn"].dtype == object

    def test_total_charges_coercion(self, mock_raw_csv):
        df = load_raw_data(str(mock_raw_csv))
        assert pd.api.types.is_float_dtype(df["total_charges"])
        # Row with blank total_charges should be NaN
        assert df["total_charges"].isna().sum() == 1
