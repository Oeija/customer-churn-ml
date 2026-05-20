from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from customer_churn_ml.utils.logger import get_logger
from customer_churn_ml.utils.metrics import evaluate_at_threshold, threshold_sweep

logger = get_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a trained model on the test set at a fixed threshold.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Preprocessed test features.
        y_test: Ground-truth test labels.
        threshold: Probability cutoff for positive class.

    Returns:
        Dictionary with precision, recall, f1, roc_auc,
        and the raw classification report string.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = evaluate_at_threshold(y_test, proba, threshold)
    report = classification_report(y_test, y_pred, digits=3)

    logger.info(
        "Evaluation (threshold=%.2f) — precision=%.3f, recall=%.3f, f1=%.3f, roc_auc=%.3f",
        threshold, metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"],
    )

    return {
        "threshold": threshold,
        **metrics,
        "classification_report": report,
    }


def sweep_thresholds(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: list,
) -> Dict[str, Any]:
    """Run a threshold sweep and return the results table + best F1 threshold.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Preprocessed test features.
        y_test: Ground-truth test labels.
        thresholds: List of probability cutoffs to evaluate.

    Returns:
        Dictionary with sweep_df (DataFrame) and best_f1_threshold.
    """
    proba = model.predict_proba(X_test)[:, 1]
    sweep_df = threshold_sweep(y_test, proba, thresholds)

    best_idx = sweep_df["F1"].idxmax()
    best_thresh = sweep_df.loc[best_idx, "Threshold"]

    logger.info(
        "Best F1=%.3f at threshold=%.2f",
        sweep_df.loc[best_idx, "F1"], best_thresh,
    )

    return {
        "sweep_df": sweep_df,
        "best_f1_threshold": best_thresh,
    }
