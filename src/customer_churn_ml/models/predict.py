import os
from pathlib import Path
from typing import Any, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifacts(
    model_dir: Union[str, Path],
) -> Tuple[Pipeline, Any]:
    """Load the preprocessor and trained model from disk.

    Expects the directory to contain:
        - preprocessor.joblib (fitted ColumnTransformer)
        - model.ubj or model.json (XGBoost raw model)

    Args:
        model_dir: Path to the local artifact directory.

    Returns:
        Tuple of (preprocessor, model).
    """
    model_dir = Path(model_dir)

    preprocessor_path = model_dir / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    model_path = model_dir / "model.ubj"
    if not model_path.exists():
        model_path = model_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_dir}")

    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor from %s", preprocessor_path)

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model._estimator_type = "classifier"
        model.load_model(str(model_path))
        logger.info("Loaded XGBoost model from %s", model_path)
    except Exception:
        # Fallback: assume sklearn model
        model = joblib.load(model_path)
        logger.info("Loaded sklearn model from %s", model_path)

    return preprocessor, model


def predict(
    df: pd.DataFrame,
    preprocessor: Pipeline,
    model: Any,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Generate churn predictions for a new DataFrame.

    Args:
        df: Raw features (before preprocessing). Must contain all expected columns.
        preprocessor: Fitted ColumnTransformer or sklearn Pipeline.
        model: Fitted classifier with predict_proba.
        threshold: Probability cutoff for flagging churn.

    Returns:
        DataFrame with original columns + churn_proba and churn_flag.
    """
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[:, 1]
    flag = (proba >= threshold).astype(int)

    result = df.copy()
    result["churn_proba"] = np.round(proba, 4)
    result["churn_flag"] = flag

    logger.info(
        "Predicted on %d rows. Predicted churners: %d (threshold=%.2f).",
        len(result), flag.sum(), threshold,
    )
    return result
