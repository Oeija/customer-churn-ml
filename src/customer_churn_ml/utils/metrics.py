from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute precision, recall, F1 and ROC-AUC at a given probability threshold.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Probability cutoff.
        pos_label: Label considered positive.

    Returns:
        Dictionary with precision, recall, f1, roc_auc.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float],
    pos_label: int = 1,
) -> pd.DataFrame:
    """Evaluate model across a list of thresholds.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities for the positive class.
        thresholds: List of probability cutoffs to evaluate.
        pos_label: Label considered positive.

    Returns:
        DataFrame with one row per threshold and columns
        Threshold, Precision, Recall, F1.
    """
    rows = []
    for thresh in thresholds:
        metrics = evaluate_at_threshold(y_true, y_proba, thresh, pos_label)
        rows.append(
            {
                "Threshold": thresh,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    digits: int = 3,
) -> None:
    """Pretty-print a sklearn classification report."""
    print(classification_report(y_true, y_pred, digits=digits))
