"""FastAPI application for real-time churn prediction."""

import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from customer_churn_ml.app.explain import (
    explain_prediction,
    load_feature_names,
    load_shap_explainer,
)
from customer_churn_ml.app.schemas import (
    BatchPredictionResponse,
    CustomerFeatures,
    ExplainabilityResponse,
    PredictionResponse,
    RecommendationItem,
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
    "shap_explainer": None,
    "feature_names": None,
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
    """Load preprocessor + model + SHAP explainer on startup; clear on shutdown."""
    config = load_config(validate_paths=False)
    _artifacts["config"] = config
    artifacts_dir = config["paths"]["artifacts_dir"]
    _artifacts["threshold"] = config["models"]["threshold"]

    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path_ubj = os.path.join(artifacts_dir, "model.ubj")
    model_path_json = os.path.join(artifacts_dir, "model.json")

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
            "Model not found in %s. POST /predict will return 503.", artifacts_dir
        )

    # --- SHAP Explainer ---
    if _artifacts["model"] is not None:
        _artifacts["shap_explainer"] = load_shap_explainer(_artifacts["model"])
        _artifacts["feature_names"] = load_feature_names(artifacts_dir)
    else:
        logger.warning("Model not loaded; skipping SHAP explainer initialization.")

    yield
    _artifacts.clear()


app = FastAPI(
    title="Churney API",
    description="Real-time churn probability scoring for telecom customers.",
    version="0.1.0",
    lifespan=lifespan,
)


def _check_artifacts():
    """Raise 503 if model artifacts were not loaded on startup."""
    if _artifacts["preprocessor"] is None or _artifacts["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run scripts/train.py first to generate them.",
        )


@app.get("/")
async def root() -> dict:
    """API root — returns basic info and available endpoints."""
    return {
        "name": app.title,
        "version": app.version,
        "description": app.description,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    healthy = _artifacts["preprocessor"] is not None and _artifacts["model"] is not None
    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": healthy,
        "threshold": _artifacts["threshold"],
        "explainability_ready": _artifacts["shap_explainer"] is not None,
    }


def _preprocess_customers(df: pd.DataFrame) -> np.ndarray:
    """Apply feature engineering, binary encoding, and preprocessing."""
    from customer_churn_ml.features.build_features import build_features
    from customer_churn_ml.data.preprocess import encode_binary_columns

    config = _artifacts["config"]
    df = build_features(df, config)
    binary_map = config["features"].get("binary_map", {})
    df = encode_binary_columns(df, binary_map)
    return _artifacts["preprocessor"].transform(df), df


def _make_prediction(
    customer_features: CustomerFeatures,
    explain: bool = False,
) -> PredictionResponse:
    """Predict churn for a single customer and optionally generate recommendations."""
    # Convert Pydantic object -> DataFrame (single row)
    raw_df = pd.DataFrame([customer_features.model_dump()])

    # Preprocess
    X, df_fe = _preprocess_customers(raw_df)
    X_row = X[0]

    # Predict
    proba = _artifacts["model"].predict_proba(X)[:, 1][0]
    threshold = _artifacts["threshold"]
    churn_flag = int(proba >= threshold)

    recommendations = None
    if explain and churn_flag == 1:
        explainer = _artifacts["shap_explainer"]
        feature_names = _artifacts["feature_names"]
        if explainer is not None and feature_names is not None:
            explain_result = explain_prediction(
                explainer,
                feature_names,
                X_row,
                raw_df_row=df_fe.iloc[0],
                top_n=3,
            )
            if explain_result:
                recommendations = [
                    RecommendationItem(**rec)
                    for rec in explain_result["recommendations"]
                ]

    return PredictionResponse(
        churn_proba=round(float(proba), 4),
        churn_flag=churn_flag,
        recommendations=recommendations,
    )


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(
    customers: List[CustomerFeatures],
    explain: bool = Query(
        False, description="Include SHAP-based recommendations for churners"
    ),
) -> BatchPredictionResponse:
    """Predict churn probability for a batch of customers.

    Accepts a list of CustomerFeatures and returns predicted probabilities
    and binary churn flags for each record.

    Set `explain=true` to receive business recommendations for flagged churners.
    """
    if not customers:
        raise HTTPException(status_code=422, detail="Empty customer list provided.")

    _check_artifacts()

    predictions = [_make_prediction(c, explain=explain) for c in customers]

    churn_count = sum(1 for p in predictions if p.churn_flag == 1)
    logger.info(
        "Predicted on %d rows | churners flagged: %d (threshold=%.2f) | explain=%s",
        len(predictions),
        churn_count,
        _artifacts["threshold"],
        explain,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        threshold=_artifacts["threshold"],
    )


@app.post("/explain", response_model=List[ExplainabilityResponse])
async def explain(customers: List[CustomerFeatures]) -> List[ExplainabilityResponse]:
    """Generate SHAP explainability breakdowns for a batch of customers.

    Returns the top churn-driving features and business recommendations
    for each record, regardless of churn flag.
    """
    if not customers:
        raise HTTPException(status_code=422, detail="Empty customer list provided.")

    _check_artifacts()

    explainer = _artifacts["shap_explainer"]
    feature_names = _artifacts["feature_names"]
    if explainer is None or feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability artifacts not loaded. Ensure feature_names.json and shap are available.",
        )

    results: List[ExplainabilityResponse] = []

    for customer in customers:
        raw_df = pd.DataFrame([customer.model_dump()])
        X, df_fe = _preprocess_customers(raw_df)
        X_row = X[0]
        proba = _artifacts["model"].predict_proba(X)[:, 1][0]
        threshold = _artifacts["threshold"]
        churn_flag = int(proba >= threshold)

        explain_result = explain_prediction(
            explainer,
            feature_names,
            X_row,
            raw_df_row=df_fe.iloc[0],
            top_n=3,
        )

        top_features = explain_result["top_features"] if explain_result else []
        recommendations = explain_result["recommendations"] if explain_result else []

        # Convert top_features tuples to dicts for the response
        top_features_dicts = [
            {
                "feature": f[0],
                "display_name": f[0],  # Will be overridden by schema if needed
                "shap_value": round(f[1], 6),
                "feature_value": f[2],
                "recommendation": rec["recommendation"],
            }
            for f, rec in zip(top_features, recommendations)
        ]

        results.append(
            ExplainabilityResponse(
                churn_proba=round(float(proba), 4),
                churn_flag=churn_flag,
                top_features=top_features_dicts,
                threshold=threshold,
            )
        )

    logger.info(
        "Explained %d rows | explainability_ready=%s",
        len(results),
        explainer is not None,
    )

    return results
