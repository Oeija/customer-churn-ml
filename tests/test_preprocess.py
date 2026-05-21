"""Unit tests for preprocessing pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.data.preprocess import (
    build_preprocessor,
    encode_binary_columns,
    preprocess_splits,
)
from customer_churn_ml.utils.config import load_config


@pytest.fixture
def dummy_config():
    return load_config()


@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "partner": ["Yes", "No", "Yes"],
        "dependents": ["No", "No", "Yes"],
        "phone_service": ["Yes", "No", "Yes"],
        "paperless_billing": ["Yes", "Yes", "No"],
        "tenure": [1, 34, 2],
        "monthly_charges": [29.85, 56.95, 53.85],
        "total_charges": [29.85, 1889.5, 108.15],
        "senior_citizen": [0, 0, 0],
        "multiple_lines": ["No", "No", "Yes"],
        "internet_service": ["DSL", "DSL", "Fiber optic"],
        "online_security": ["No", "Yes", "No"],
        "online_backup": ["No", "Yes", "No"],
        "device_protection": ["No", "Yes", "No"],
        "tech_support": ["No", "No", "Yes"],
        "streaming_tv": ["No", "No", "No"],
        "streaming_movies": ["No", "No", "No"],
        "contract": ["Month-to-month", "One year", "Month-to-month"],
        "payment_method": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
    })


class TestPreprocess:
    def test_encode_binary_columns(self, dummy_df, dummy_config):
        binary_map = dummy_config["features"]["binary_map"]
        result = encode_binary_columns(dummy_df, binary_map)
        assert result["gender"].dtype.name == "Int64"
        assert set(result["gender"].dropna().unique()).issubset({0, 1})
        assert set(result["partner"].dropna().unique()).issubset({0, 1})

    def test_build_preprocessor(self, dummy_config):
        preprocessor = build_preprocessor(dummy_config)
        assert preprocessor is not None
        # Use .transformers (constructor arg) instead of .named_transformers_ (fitted property)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names
        assert "bin" in transformer_names

    def test_preprocess_splits_shape(self, dummy_df, dummy_config):
        # Duplicate for train/test since we just care about shapes
        X_train = dummy_df.copy()
        X_test = dummy_df.copy()
        X_train_proc, X_test_proc, feature_names = preprocess_splits(
            X_train, X_test, dummy_config
        )
        assert X_train_proc.shape[0] == 3
        assert X_test_proc.shape[0] == 3
        assert X_train_proc.shape[1] == len(feature_names)
        assert X_train_proc.shape[1] == X_test_proc.shape[1]

    def test_preprocess_no_nan(self, dummy_df, dummy_config):
        X_train = dummy_df.copy()
        X_test = dummy_df.copy()
        X_train_proc, X_test_proc, _ = preprocess_splits(
            X_train, X_test, dummy_config
        )
        # Use pandas isnull to handle mixed dtypes (int/float from OneHotEncoder + StandardScaler)
        assert not pd.isnull(X_train_proc).any()
        assert not pd.isnull(X_test_proc).any()

    def test_feature_names_count(self, dummy_df, dummy_config):
        X_train = dummy_df.copy()
        X_test = dummy_df.copy()
        _, _, feature_names = preprocess_splits(X_train, X_test, dummy_config)
        # 4 numeric + 10 categorical with drop_first -> variable one-hot count + 5 binary
        assert len(feature_names) > 0
        assert "tenure" in feature_names
        assert "senior_citizen" in feature_names
