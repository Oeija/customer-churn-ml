import numpy as np
import pandas as pd

from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def add_tenure_group(
    df: pd.DataFrame,
    bins: list,
    labels: list,
    include_lowest: bool = True,
    right: bool = True,
) -> pd.DataFrame:
    """Bin`tenure into lifecycle stage groups.

    Args:
        df: DataFrame containing a tenure column.
        bins: Edge values for the bins.
        labels: String label for each bin.
        include_lowest: Whether the first interval is closed.
        right: Whether bins are right-inclusive.

    Returns:
        DataFrame with new tenure_group column.
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=include_lowest,
        right=right,
    )
    logger.info("Added tenure_group (bins=%s, labels=%s).", bins, labels)
    return df


def add_num_services(
    df: pd.DataFrame,
    service_columns: list,
) -> pd.DataFrame:
    """Count how many premium add-on services a customer has.

    Args:
        df: DataFrame with service columns (values "Yes" / "No").
        service_columns: List of column names to count.

    Returns:
        DataFrame with new num_services column.
    """
    df = df.copy()
    df["num_services"] = (df[service_columns] == "Yes").sum(axis=1).astype(int)
    logger.info("Added num_services from columns %s.", service_columns)
    return df


def add_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate average monthly charge (total_charges / tenure).

    For customers with tenure == 0, the value is set to monthly_charges
    (their first-month charge is the best available proxy).

    Args:
        df: DataFrame with total_charges and tenure columns.

    Returns:
        DataFrame with new avg_monthly_charge column.
    """
    df = df.copy()
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["total_charges"] / df["tenure"],
        df["monthly_charges"],  # proxy for tenure=0 customers
    )
    logger.info("Added avg_monthly_charge (proxy used for tenure=0).")
    return df


def add_has_internet(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag indicating whether the customer has internet service."""
    df = df.copy()
    df["has_internet"] = (df["internet_service"] != "No").astype(int)
    logger.info("Added has_internet.")
    return df


def add_has_phone(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag indicating whether the customer has phone service."""
    df = df.copy()
    df["has_phone"] = (df["phone_service"] == "Yes").astype(int)
    logger.info("Added has_phone.")
    return df


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the full feature engineering pipeline.

    Steps (order matters):
        1. tenure_group - lifecycle stage bins
        2. num_services - count of premium add-ons
        3. avg_monthly_charge - historical average spend
        4. has_internet - internet subscription flag
        5. has_phone - phone subscription flag

    Args:
        df: Cleaned DataFrame from ingestion (before train/test split).
        config: Project configuration dict.

    Returns:
        DataFrame with all derived features appended.
    """
    fe_cfg = config.get("feature_engineering", {})

    # 1. Tenure group
    tenure_cfg = fe_cfg.get("tenure_bins", {})
    if tenure_cfg:
        df = add_tenure_group(
            df,
            bins=tenure_cfg["bins"],
            labels=tenure_cfg["labels"],
            include_lowest=tenure_cfg.get("include_lowest", True),
            right=tenure_cfg.get("right", True),
        )

    # 2. Number of services
    service_cols = fe_cfg.get("service_columns", [])
    if service_cols:
        df = add_num_services(df, service_cols)

    # 3. Average monthly charge
    if "total_charges" in df.columns and "tenure" in df.columns:
        df = add_avg_monthly_charge(df)

    # 4 & 5. Boolean flags
    if fe_cfg.get("add_has_internet", True):
        df = add_has_internet(df)
    if fe_cfg.get("add_has_phone", True):
        df = add_has_phone(df)

    logger.info(
        "Feature engineering complete. Shape: %s, new columns: %s",
        df.shape,
        [
            c
            for c in df.columns
            if c
            not in config["features"]["numeric"]
            + config["features"]["binary"]
            + config["features"]["categorical"]
            + [config["features"]["target"]]
        ],
    )
    return df
