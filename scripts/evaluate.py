"""Evaluation script: load saved artifacts and evaluate on test data.

Usage:
    python -m scripts.evaluate [--fresh-split] [--data data/raw/telco-customer-churn-raw.csv]
"""

import argparse
import sys
from datetime import datetime
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


def main(data_path: str = None, fresh_split: bool = False):
    config = load_config()
    paths = config["paths"]
    artifacts_dir = Path(paths["artifacts_dir"])
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    model_path_ubj = artifacts_dir / "model.ubj"
    model_path_json = artifacts_dir / "model.json"

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
        raise FileNotFoundError(f"Model not found in {artifacts_dir}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    test_features_path = processed_dir / "test_features.parquet"
    test_labels_path = processed_dir / "test_labels.parquet"

    if not fresh_split and test_features_path.exists() and test_labels_path.exists():
        logger.info("Loading processed test data from %s", processed_dir)
        X_test = pd.read_parquet(test_features_path)
        y_test = pd.read_parquet(test_labels_path)["churn"]
        # Apply binary encoding only (feature engineering already done)
        from customer_churn_ml.data.preprocess import encode_binary_columns
        binary_map = config["features"].get("binary_map", {})
        X_test = encode_binary_columns(X_test, binary_map)
    else:
        if not fresh_split and not test_features_path.exists():
            logger.warning(
                "Processed test data not found at %s. Falling back to fresh split from raw data.",
                processed_dir,
            )
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

    report_lines = [
        "=" * 60,
        "Evaluation Results",
        "=" * 60,
        f"Threshold:  {threshold}",
        f"Precision:  {eval_results['precision']:.3f}",
        f"Recall:     {eval_results['recall']:.3f}",
        f"F1:         {eval_results['f1']:.3f}",
        f"ROC-AUC:    {eval_results['roc_auc']:.3f}",
        "",
        "Classification Report:",
        eval_results["classification_report"],
    ]

    thresholds = config["models"].get("thresholds_to_evaluate", [])
    if thresholds:
        sweep = sweep_thresholds(model, X_test_proc, y_test.values, thresholds)
        report_lines.extend([
            "",
            "Threshold Sweep:",
            sweep["sweep_df"].to_string(index=False),
        ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = outputs_dir / f"evaluation_{timestamp}.txt"
    eval_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved evaluation report to %s", eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved churn model.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw CSV (defaults to config raw_data path).",
    )
    parser.add_argument(
        "--fresh-split",
        action="store_true",
        default=False,
        help="Force a fresh train/test split from raw data instead of using the saved processed split.",
    )
    args = parser.parse_args()
    main(data_path=args.data, fresh_split=args.fresh_split)
