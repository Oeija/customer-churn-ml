import re

import pandas as pd

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert camelCase / PascalCase column names to snake_case
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = [re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", c) for c in df.columns]
    df.columns = [re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c).lower() for c in df.columns]
    return df


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV and perform basic cleaning.

    Steps:
        1. Read CSV.
        2. Strip whitespace and convert column names to snake_case.
        3. Coerce total_charges to numeric (blanks become NaN).
        4. Drop the non-predictive customer_id column if present.

    Args:
        path: Absolute or relative path to the raw CSV file.

    Returns:
        Cleaned DataFrame ready for validation / preprocessing.
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    df = standardise_columns(df)

    # Coerce total_charges = the dataset stores blanks as strings for new customers
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        blank_count = df["total_charges"].isna().sum()
        if blank_count:
            logger.info(
                "Coerced %d blank 'total_charges' values to NaN (tenure=0 customers).",
                blank_count,
            )

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
        logger.info("Dropped non-predictive column 'customer_id'.")

    logger.info("Loaded DataFrame shape: %s", df.shape)
    return df
