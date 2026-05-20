"""FastAPI application for real-time churn prediction."""

import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from customer_churn_ml.app.schemas import (
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)
from customer_churn_ml.utils.config import load_config
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global state for loaded artifacts (populated on startup)
# ---------------------------------------------------------------------------
_artifacts = {
    "preprocessor": None,
    "model": None,
    "threshold": 0.30,
    "config": None,
}


def _load_xgb_model(model_path: str):
    """Load an XGBoost model from UBJ or JSON format."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for serving. Install with: uv pip install 'xgboost>=2.0,<3.0'"
        ) from exc
    model = XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model(model_path)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load preprocessor + model on startup; clear on shutdown."""
    config = load_config()
    _artifacts["config"] = config
    model_dir = config["paths"]["serving_model_dir"]
    _artifacts["threshold"] = config["models"]["threshold"]

    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    model_path_ubj = os.path.join(model_dir, "model.ubj")
    model_path_json = os.path.join(model_dir, "model.json")

    # --- Preprocessor ---
    if os.path.exists(preprocessor_path):
        _artifacts["preprocessor"] = joblib.load(preprocessor_path)
        logger.info("Loaded preprocessor from %s", preprocessor_path)
    else:
        logger.warning(
            "Preprocessor not found at %s. POST /predict will return 503.",
            preprocessor_path,
        )

    # --- Model ---
    if os.path.exists(model_path_ubj):
        _artifacts["model"] = _load_xgb_model(model_path_ubj)
        logger.info("Loaded model from %s", model_path_ubj)
    elif os.path.exists(model_path_json):
        _artifacts["model"] = _load_xgb_model(model_path_json)
        logger.info("Loaded model from %s", model_path_json)
    else:
        logger.warning(
            "Model not found in %s. POST /predict will return 503.", model_dir
        )

    yield
    _artifacts.clear()


app = FastAPI(
    title="Churney API",
    description="Real-time churn probability scoring for telecom customers.",
    version="0.1.0",
    lifespan=lifespan,
)


def _check_artifacts():
    """Raise 503 if artifacts were not loaded on startup."""
    if _artifacts["preprocessor"] is None or _artifacts["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run scripts/train.py first to generate them.",
        )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    healthy = (
        _artifacts["preprocessor"] is not None and _artifacts["model"] is not None
    )
    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": healthy,
        "threshold": _artifacts["threshold"],
    }


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(customers: List[CustomerFeatures]) -> BatchPredictionResponse:
    """Predict churn probability for a batch of customers.

    Accepts a list of CustomerFeatures and returns predicted probabilities
    and binary churn flags for each record.
    """
    if not customers:
        raise HTTPException(status_code=422, detail="Empty customer list provided.")

    _check_artifacts()

    # Convert Pydantic objects -> DataFrame (raw features, same format as CSV)
    df = pd.DataFrame([c.model_dump() for c in customers])

    # Feature engineering + binary encoding (preprocessor was fitted on FE data)
    from customer_churn_ml.features.build_features import build_features
    from customer_churn_ml.data.preprocess import encode_binary_columns

    config = _artifacts["config"]
    df = build_features(df, config)
    binary_map = config["features"].get("binary_map", {})
    df = encode_binary_columns(df, binary_map)

    # Preprocess (ColumnTransformer handles NaN in total_charges via imputation)
    X = _artifacts["preprocessor"].transform(df)

    # Predict
    proba = _artifacts["model"].predict_proba(X)[:, 1]
    threshold = _artifacts["threshold"]
    flags = (proba >= threshold).astype(int)

    predictions = [
        PredictionResponse(churn_proba=round(float(p), 4), churn_flag=int(f))
        for p, f in zip(proba, flags)
    ]

    logger.info(
        "Predicted on %d rows | churners flagged: %d (threshold=%.2f)",
        len(predictions),
        int(flags.sum()),
        threshold,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        threshold=threshold,
    )
