import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Return the ratio of negative to positive samples for XGBoost."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        raise ValueError("Training set contains no positive samples.")
    return neg / pos


def build_classifier(model_name: str, config: dict, y_train: np.ndarray) -> Any:
    """Instantiate a classifier from config with appropriate class imbalance handling.

    Args:
        model_name: One of random_forest, lightgbm, xgboost.
        config: Project configuration dict.
        y_train: Training labels (needed to compute scale_pos_weight).

    Returns:
        Unfitted classifier instance.
    """
    model_cfg = config["models"]

    if model_name == "random_forest":
        params = model_cfg["random_forest"].copy()
        clf = RandomForestClassifier(**params)

    elif model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required. Install with: uv pip install 'lightgbm>=4.0,<5.0'"
            ) from exc
        params = model_cfg["lightgbm"].copy()
        clf = lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install with: uv pip install 'xgboost>=2.0,<3.0'"
            ) from exc
        params = model_cfg["xgboost"].copy()
        params["scale_pos_weight"] = _compute_scale_pos_weight(y_train)
        clf = XGBClassifier(**params)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    logger.info("Built classifier: %s", model_name)
    return clf


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> Tuple[Any, Dict[str, Any]]:
    """Train classifier(s) according to the active list in config.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        config: Project configuration dict.

    Returns:
        Tuple of (trained_classifier, metadata_dict).
        If multiple models are active, only the **last** one is returned
        (the orchestration script can loop externally if needed).
    """
    active_models = config["models"]["active"]
    if not active_models:
        raise ValueError("No active models specified in config.")

    trained_model = None
    metadata = {}

    for model_name in active_models:
        logger.info("Training %s...", model_name)
        clf = build_classifier(model_name, config, y_train)

        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        metadata[model_name] = {
            "train_time_seconds": round(train_time, 2),
            "n_samples": len(y_train),
            "n_features": X_train.shape[1],
            "pos_weight": getattr(clf, "scale_pos_weight", None),
        }

        logger.info(
            "%s trained in %.2fs (samples=%d, features=%d).",
            model_name,
            train_time,
            len(y_train),
            X_train.shape[1],
        )
        trained_model = clf

    return trained_model, metadata
