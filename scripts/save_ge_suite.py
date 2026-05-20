"""Persist Great Expectations expectation suite to JSON for version control."""

import json
import sys
from pathlib import Path

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
)

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from customer_churn_ml.utils.config import load_config


def build_and_save_suite(output_dir: str = "great_expectations/expectations"):
    """Build the Telco expectation suite and save as JSON."""
    config = load_config()
    features_cfg = config["features"]

    expected_columns = (
        features_cfg.get("numeric", [])
        + features_cfg.get("binary", [])
        + features_cfg.get("categorical", [])
        + [features_cfg.get("target", "churn")]
    )
    categorical_expectations = features_cfg.get("categorical_expected_values", {})

    context = gx.get_context()
    try:
        suite = context.suites.get(name="telco_expectation_suite")
    except gx.exceptions.DataContextError:
        suite = ExpectationSuite(name="telco_expectation_suite")

    for col in expected_columns:
        suite.add_expectation(ExpectColumnToExist(column=col))

    for col, values in categorical_expectations.items():
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(column=col, value_set=values)
        )

    if "churn" in categorical_expectations:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="churn"))

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "telco_expectation_suite.json"

    with open(out_path, "w") as f:
        json.dump(suite.to_json_dict(), f, indent=2)

    print(f"Saved expectation suite to {out_path}")
    print(f"Total expectations: {len(suite.expectations)}")


if __name__ == "__main__":
    build_and_save_suite()
