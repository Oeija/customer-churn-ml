from typing import Dict, Tuple

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
)

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def build_expectation_suite(
    suite_name: str = "telco_expectation_suite",
    expected_columns: list = None,
    categorical_expectations: dict = None,
) -> ExpectationSuite:
    """Build a minimal GE expectation suite for the dataset.

    Args:
        suite_name: Name of the expectation suite.
        expected_columns: List of columns that must exist.
        categorical_expectations: Mapping column -> list_of_expected_values.

    Returns:
        Configured ExpectationSuite.
    """
    context = gx.get_context()

    # Try to retrieve existing suite; create fresh if missing
    try:
        suite = context.suites.get(name=suite_name)
        logger.info("Loaded existing expectation suite: %s", suite_name)
    except gx.exceptions.DataContextError:
        suite = ExpectationSuite(name=suite_name)
        logger.info("Created new expectation suite: %s", suite_name)

    if expected_columns:
        for col in expected_columns:
            suite.add_expectation(ExpectColumnToExist(column=col))

    if categorical_expectations:
        for col, values in categorical_expectations.items():
            suite.add_expectation(
                ExpectColumnValuesToBeInSet(column=col, value_set=values)
            )

    # Target must not be null
    if categorical_expectations and "churn" in categorical_expectations:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="churn"))

    # Persist so it can be reused across validations
    try:
        context.suites.add(suite)
    except gx.exceptions.DataContextError:
        # Already exists, overwrite by re-adding with same name
        pass

    return suite


def validate_data(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[bool, Dict]:
    """Run Great Expectations validation on a DataFrame.

    Args:
        df: DataFrame to validate.
        config: Project configuration dict (from load_config).

    Returns:
        Tuple of (success: bool, results: dict).
    """
    features_cfg = config.get("features", {})
    expected_columns = (
        features_cfg.get("numeric", [])
        + features_cfg.get("binary", [])
        + features_cfg.get("categorical", [])
        + [features_cfg.get("target", "churn")]
    )
    categorical_expectations = features_cfg.get("categorical_expected_values", {})

    suite = build_expectation_suite(
        suite_name="telco_expectation_suite",
        expected_columns=expected_columns,
        categorical_expectations=categorical_expectations,
    )

    context = gx.get_context()
    datasource_name = "pandas_datasource"
    asset_name = "telco_asset"
    batch_def_name = "telco_batch_def"

    # Ensure datasource / asset / batch_definition exist
    try:
        datasource = context.data_sources.get(datasource_name)
    except (KeyError, gx.exceptions.DataContextError):
        datasource = context.data_sources.add_pandas(name=datasource_name)

    try:
        data_asset = datasource.get_asset(asset_name)
    except (KeyError, LookupError, gx.exceptions.DataContextError):
        data_asset = datasource.add_dataframe_asset(name=asset_name)

    try:
        batch_definition = data_asset.get_batch_definition(batch_def_name)
    except (KeyError, LookupError, gx.exceptions.DataContextError):
        batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_def_name)

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    results = batch.validate(suite)

    success = results.success

    if success:
        logger.info("GE validation passed (%d expectations checked).", len(results.results))
    else:
        failed = [
            r.expectation_config.type
            for r in results.results
            if not r.success
        ]
        logger.warning("GE validation failed. Failed expectations: %s", failed)

    result_dict = {
        "success": success,
        "evaluated_expectations": results.statistics["evaluated_expectations"],
        "successful_expectations": results.statistics["successful_expectations"],
        "unsuccessful_expectations": results.statistics["unsuccessful_expectations"],
        "failed_expectations": [
            {
                "type": r.expectation_config.type,
                "column": r.expectation_config.kwargs.get("column"),
                "unexpected_percent": r.result.get("unexpected_percent"),
            }
            for r in results.results
            if not r.success
        ],
    }

    return success, result_dict
