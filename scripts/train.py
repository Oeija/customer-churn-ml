"""End-to-end training orchestration script.

Pipeline:
    config -> ingest -> validate -> split -> preprocess -> (optuna tune) -> train -> evaluate
    -> SHAP -> MLflow log -> save local artifacts

Usage:
    cd /path/to/project
    python -m scripts.train [--tune]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure src is on path even when run as __main__
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.data.ingestion import load_raw_data
from customer_churn_ml.data.preprocess import preprocess_splits
from customer_churn_ml.data.validation import validate_data
from customer_churn_ml.models.evaluate import evaluate_model, sweep_thresholds
from customer_churn_ml.models.train import train_model
from customer_churn_ml.models.tune import tune_xgboost
from customer_churn_ml.utils.config import load_config
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def _log_shap_summary(model, X_test_proc, feature_names, model_name="model"):
    """Generate and save a SHAP summary plot; return the temp file path."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Skipping SHAP plot.")
        return None

    logger.info("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_proc)

    # TreeExplainer returns list for binary classifiers [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # use positive class

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names, show=False)
    tmp_path = tempfile.mktemp(suffix="_shap_summary.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary plot saved to %s", tmp_path)
    return tmp_path


def main(tune: bool = False):
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training pipeline (tune=%s)", tune)
    logger.info("=" * 60)

    config = load_config()
    paths = config["paths"]
    model_cfg = config["models"]

    # ------------------------------------------------------------------
    # 2. Ingest raw data
    # ------------------------------------------------------------------
    df = load_raw_data(paths["raw_data"])

    # ------------------------------------------------------------------
    # 3. Feature engineering (before validation so derived columns exist)
    # ------------------------------------------------------------------
    from customer_churn_ml.features.build_features import build_features
    df = build_features(df, config)
    logger.info("Feature engineering complete. Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # 4. Validate with Great Expectations
    # ------------------------------------------------------------------
    logger.info("Running data validation...")
    valid, val_results = validate_data(df, config)
    if not valid:
        logger.error("Data validation failed. Aborting training.")
        logger.error("Failures: %s", val_results["failed_expectations"])
        sys.exit(1)
    logger.info("Data validation passed.")

    # ------------------------------------------------------------------
    # 4. Train / test split (before any preprocessing)
    # ------------------------------------------------------------------
    target_col = config["features"]["target"]
    y = df[target_col].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col])

    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", True) else None,
    )
    logger.info(
        "Split — train: %s, test: %s", X_train.shape, X_test.shape
    )

    # ------------------------------------------------------------------
    # 5. Preprocess (feature engineering → encode → ColumnTransformer)
    # ------------------------------------------------------------------
    logger.info("Building preprocessing pipeline...")
    X_train_proc, X_test_proc, feature_names = preprocess_splits(
        X_train, X_test, config
    )

    # ------------------------------------------------------------------
    # 6. (Optional) Optuna hyperparameter tuning
    # ------------------------------------------------------------------
    if tune and "xgboost" in model_cfg["active"]:
        logger.info("Running Optuna hyperparameter tuning...")
        tune_results = tune_xgboost(X_train_proc, y_train.values, config, n_trials=30)
        best_params = tune_results["best_params"]
        logger.info(
            "Best recall=%.4f | Params=%s", tune_results["best_recall"], best_params
        )
        # Override config for training
        model_cfg["xgboost"] = {k: v for k, v in best_params.items() if k not in ("scale_pos_weight", "eval_metric", "random_state", "n_jobs")}

    # ------------------------------------------------------------------
    # 7. Train model(s)
    # ------------------------------------------------------------------
    logger.info("Training model(s): %s", model_cfg["active"])
    model, train_meta = train_model(X_train_proc, y_train.values, config)

    # ------------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------------
    threshold = model_cfg["threshold"]
    eval_results = evaluate_model(model, X_test_proc, y_test.values, threshold)
    logger.info("\n%s", eval_results["classification_report"])

    thresholds = model_cfg.get("thresholds_to_evaluate", [])
    if thresholds:
        sweep = sweep_thresholds(model, X_test_proc, y_test.values, thresholds)
        logger.info(
            "\nThreshold sweep:\n%s", sweep["sweep_df"].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 9. SHAP explainability
    # ------------------------------------------------------------------
    shap_plot_path = _log_shap_summary(model, X_test_proc, feature_names)

    # ------------------------------------------------------------------
    # 10. MLflow logging
    # ------------------------------------------------------------------
    logger.info("Logging to MLflow...")
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    mlflow.set_tracking_uri(config["mlflow"].get("mlruns_uri", "file://./mlruns"))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Parameters
        mlflow.log_params(train_meta)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("preprocessor", "ColumnTransformer with SimpleImputer + StandardScaler + OneHotEncoder + FeatureEngineering")
        mlflow.log_param("n_features", len(feature_names))
        if tune:
            mlflow.log_param("optuna_tuned", True)

        # Metrics
        mlflow.log_metric("precision", eval_results["precision"])
        mlflow.log_metric("recall", eval_results["recall"])
        mlflow.log_metric("f1", eval_results["f1"])
        mlflow.log_metric("roc_auc", eval_results["roc_auc"])

        # SHAP artifact
        if shap_plot_path and os.path.exists(shap_plot_path):
            mlflow.log_artifact(shap_plot_path, artifact_path="shap")
            os.remove(shap_plot_path)

        # Full pipeline artifact
        from customer_churn_ml.data.preprocess import build_preprocessor, encode_binary_columns
        from sklearn.pipeline import Pipeline as SklearnPipeline

        X_train_enc = encode_binary_columns(X_train, config["features"]["binary_map"])
        preprocessor = build_preprocessor(config)
        preprocessor.fit(X_train_enc)

        full_pipeline = SklearnPipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])
        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path=config["mlflow"]["artifact_paths"]["full_pipeline"],
            pip_requirements=["scikit-learn", "xgboost==2.1.4", "pandas", "numpy"],
            conda_env=None,
        )

        # Raw XGBoost model
        model._estimator_type = "classifier"
        mlflow.xgboost.log_model(
            model,
            artifact_path=config["mlflow"]["artifact_paths"]["raw_model"],
            pip_requirements=["xgboost==2.1.4"],
            conda_env=None,
        )

        logger.info("MLflow logging complete.")

    # ------------------------------------------------------------------
    # 11. Save local artifacts for serving
    # ------------------------------------------------------------------
    model_dir = Path(paths["serving_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = model_dir / "preprocessor.joblib"
    model_path = model_dir / "model.ubj"
    config_path = model_dir / "config.yaml"
    feature_names_path = model_dir / "feature_names.json"

    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Saved preprocessor to %s", preprocessor_path)

    try:
        model.save_model(str(model_path))
        logger.info("Saved model to %s", model_path)
    except AttributeError:
        joblib.dump(model, model_path)
        logger.info("Saved sklearn model to %s", model_path)

    shutil.copy(project_root / "config" / "config.yaml", config_path)
    logger.info("Saved config copy to %s", config_path)

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Saved feature names to %s", feature_names_path)

    logger.info("=" * 60)
    logger.info("Training pipeline complete. Artifacts saved to %s", model_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run Optuna hyperparameter tuning before training (XGBoost only).",
    )
    args = parser.parse_args()
    main(tune=args.tune)
