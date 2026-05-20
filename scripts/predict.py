"""Batch prediction script: load artifacts and score a CSV file.

Usage:
    python -m scripts.predict --input data/raw/new_customers.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.data.ingestion import standardise_columns
from customer_churn_ml.models.predict import load_artifacts, predict
from customer_churn_ml.utils.config import load_config
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def main(input_path: str, output_path: str):
    config = load_config()
    model_dir = Path(config["paths"]["serving_model_dir"])

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    preprocessor, model = load_artifacts(model_dir)
    threshold = config["models"]["threshold"]

    # ------------------------------------------------------------------
    # Load and prepare new data
    # ------------------------------------------------------------------
    logger.info("Loading input data from %s", input_path)
    df = pd.read_csv(input_path)
    df = standardise_columns(df)

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Feature engineering + binary encoding (preprocessor was fitted on FE data)
    from customer_churn_ml.features.build_features import build_features
    from customer_churn_ml.data.preprocess import encode_binary_columns

    df = build_features(df, config)
    binary_map = config["features"].get("binary_map", {})
    df = encode_binary_columns(df, binary_map)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    results = predict(df, preprocessor, model, threshold=threshold)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(results), output_path)
    logger.info(
        "Flagged churners: %d (threshold=%.2f)",
        results["churn_flag"].sum(),
        threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn prediction.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CSV with customer features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to write predictions CSV.",
    )
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output)
