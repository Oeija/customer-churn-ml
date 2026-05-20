"""Evaluation script: load saved artifacts and evaluate on test data.

Usage:
    python -m scripts.evaluate --data data/raw/telco-customer-churn-raw.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.data.ingestion import load_raw_data
from customer_churn_ml.data.preprocess import preprocess_splits
from customer_churn_ml.models.evaluate import evaluate_model, sweep_thresholds
from customer_churn_ml.utils.config import load_config
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def _load_xgb_model(model_path: str):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model._estimator_type = "classifier"
    model.load_model(model_path)
    return model


def main(data_path: str = None):
    config = load_config()
    paths = config["paths"]
    model_dir = Path(paths["serving_model_dir"])

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    preprocessor_path = model_dir / "preprocessor.joblib"
    model_path_ubj = model_dir / "model.ubj"
    model_path_json = model_dir / "model.json"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor from %s", preprocessor_path)

    if model_path_ubj.exists():
        model = _load_xgb_model(str(model_path_ubj))
        logger.info("Loaded model from %s", model_path_ubj)
    elif model_path_json.exists():
        model = _load_xgb_model(str(model_path_json))
        logger.info("Loaded model from %s", model_path_json)
    else:
        raise FileNotFoundError(f"Model not found in {model_dir}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    raw_path = data_path or paths["raw_data"]
    df = load_raw_data(raw_path)

    target_col = config["features"]["target"]
    y = df[target_col].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col])

    split_cfg = config["split"]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", True) else None,
    )

    # Apply feature engineering + binary encoding, then transform with saved preprocessor
    from customer_churn_ml.features.build_features import build_features
    from customer_churn_ml.data.preprocess import encode_binary_columns

    X_test = build_features(X_test, config)
    binary_map = config["features"].get("binary_map", {})
    X_test = encode_binary_columns(X_test, binary_map)
    X_test_proc = preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    threshold = config["models"]["threshold"]
    eval_results = evaluate_model(model, X_test_proc, y_test.values, threshold)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Threshold:  {threshold}")
    print(f"Precision:  {eval_results['precision']:.3f}")
    print(f"Recall:     {eval_results['recall']:.3f}")
    print(f"F1:         {eval_results['f1']:.3f}")
    print(f"ROC-AUC:    {eval_results['roc_auc']:.3f}")
    print("\nClassification Report:")
    print(eval_results["classification_report"])

    thresholds = config["models"].get("thresholds_to_evaluate", [])
    if thresholds:
        sweep = sweep_thresholds(model, X_test_proc, y_test.values, thresholds)
        print("\nThreshold Sweep:")
        print(sweep["sweep_df"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved churn model.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw CSV (defaults to config raw_data path).",
    )
    args = parser.parse_args()
    main(data_path=args.data)
