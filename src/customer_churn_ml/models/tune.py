from typing import Any, Dict

import numpy as np
import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def _objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    threshold: float,
):
    """Return an Optuna objective function that maximises recall."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for tuning. Install with: uv pip install 'xgboost>=2.0,<3.0'"
        ) from exc

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        return recall

    return objective


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    n_trials: int = 30,
    val_size: float = 0.2,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimisation for XGBoost.

    Splits the training data again into train/validation for the search,
    so the held-out test set remains untouched.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        config: Project configuration dict.
        n_trials: Number of Optuna trials.
        val_size: Fraction of *training* data to hold out for validation during tuning.

    Returns:
        Dictionary with best_params and best_recall.
    """
    threshold = config["models"]["threshold"]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Internal train/val split for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train,
    )

    objective = _objective_factory(
        X_tr, y_tr, X_val, y_val, scale_pos_weight, threshold
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }
    )

    logger.info(
        "Optuna tuning complete. Best recall=%.4f | Params=%s",
        study.best_value,
        best_params,
    )

    return {
        "best_params": best_params,
        "best_recall": study.best_value,
        "study": study,
    }
