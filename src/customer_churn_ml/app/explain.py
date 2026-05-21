"""SHAP explainability helpers for the churn prediction API.

Handles per-user SHAP value computation, top-feature extraction,
and recommendation generation.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from customer_churn_ml.app.recommendations import build_recommendations
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def load_shap_explainer(model: Any) -> Optional[Any]:
    """Create a SHAP TreeExplainer for the loaded tree-based model.

    Args:
        model: Loaded tree-based classifier (XGBoost, LightGBM, etc.).

    Returns:
        shap.TreeExplainer instance, or None if shap is not installed
        or the model is not tree-based.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Explainability features disabled.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized successfully.")
        return explainer
    except Exception as exc:
        logger.warning(
            "Could not create TreeExplainer: %s. Explainability disabled.", exc
        )
        return None


def load_feature_names(artifacts_dir: str) -> Optional[List[str]]:
    """Load the ordered list of post-preprocessing feature names.

    Args:
        artifacts_dir: Directory containing feature_names.json.

    Returns:
        List of feature names, or None if the file is missing.
    """
    path = os.path.join(artifacts_dir, "feature_names.json")
    if not os.path.exists(path):
        logger.warning(
            "feature_names.json not found at %s. Explainability disabled.", path
        )
        return None
    with open(path, "r") as f:
        names = json.load(f)
    logger.info("Loaded %d feature names from %s", len(names), path)
    return names


def get_top_churn_features(
    shap_values_row: np.ndarray,
    feature_names: List[str],
    feature_values_row: np.ndarray,
    top_n: int = 3,
) -> List[Tuple[str, float, Any]]:
    """Extract the top N features pushing the prediction toward churn.

    Filters to only positive SHAP values (features that increase churn probability)
    and returns the largest ones.

    Args:
        shap_values_row: 1-D array of SHAP values for a single prediction.
        feature_names: Ordered list of feature names matching the SHAP array.
        feature_values_row: 1-D array of raw (preprocessed) feature values.
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, shap_value, feature_value) tuples,
        sorted by descending shap_value.
    """
    # Only consider positive SHAP values (pushing toward churn)
    positive_mask = shap_values_row > 0
    positive_indices = np.where(positive_mask)[0]

    if len(positive_indices) == 0:
        return []

    # Sort by SHAP value descending
    sorted_idx = positive_indices[np.argsort(-shap_values_row[positive_indices])]
    top_idx = sorted_idx[:top_n]

    return [
        (
            feature_names[i],
            float(shap_values_row[i]),
            feature_values_row[i],
        )
        for i in top_idx
    ]


def explain_prediction(
    explainer: Any,
    feature_names: List[str],
    X_proc: np.ndarray,
    raw_df_row: pd.Series,
    top_n: int = 3,
) -> Optional[Dict[str, Any]]:
    """Generate explainability output for a single preprocessed row.

    Args:
        explainer: shap.TreeExplainer instance.
        feature_names: Ordered list of feature names.
        X_proc: Preprocessed feature vector (1-D or 2-D with 1 row).
        raw_df_row: Raw feature values as a pandas Series (for recommendation rules).
        top_n: Number of top churn-driving features to extract.

    Returns:
        Dict with 'top_features' and 'recommendations', or None if explainer is None.
    """
    if explainer is None:
        return None

    # Ensure 2-D array for SHAP
    if X_proc.ndim == 1:
        X_proc = X_proc.reshape(1, -1)

    shap_values = explainer.shap_values(X_proc)

    # TreeExplainer returns list for binary classifiers [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = shap_values[0]  # 1-D array
    feature_values_row = X_proc[0]

    top_features = get_top_churn_features(
        shap_row, feature_names, feature_values_row, top_n=top_n
    )

    recommendations = build_recommendations(top_features, raw_df_row)

    return {
        "top_features": top_features,
        "recommendations": recommendations,
    }
