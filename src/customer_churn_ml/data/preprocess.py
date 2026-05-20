from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn_ml.features.build_features import build_features
from customer_churn_ml.utils.logger import get_logger

logger = get_logger(__name__)


def encode_binary_columns(
    df: pd.DataFrame,
    binary_map: dict,
) -> pd.DataFrame:
    """Map binary Yes/No (and gender Male/Female) columns to 0/1 integers.

    Args:
        df: DataFrame with raw categorical binary columns.
        binary_map: Nested dict from config, e.g.
            ``{"gender": {"Male": 1, "Female": 0}, ...}``.

    Returns:
        DataFrame with binary columns replaced by integers.
    """
    df = df.copy()
    for col, mapping in binary_map.items():
        if col in df.columns:
            # Capture original unexpected values (before mapping turns them into NaN)
            unmapped_unexpected = df.loc[~df[col].isin(mapping.keys()), col].unique()
            if len(unmapped_unexpected) > 0:
                # Filter out actual NaN values, they may be legitimate missing data
                unmapped_unexpected = [v for v in unmapped_unexpected if pd.notna(v)]
                if unmapped_unexpected:
                    raise ValueError(
                        f"Unexpected values in binary column '{col}': {unmapped_unexpected}. "
                        f"Expected one of {list(mapping.keys())}."
                    )
            df[col] = df[col].map(mapping).astype("Int64")  # nullable integer
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Construct a ColumnTransformer from project configuration.

    The pipeline handles three column groups:
      * **numeric** - imputation + optional scaling
      * **categorical** - one-hot encoding (drop first level)
      * **binary** - passthrough (already 0/1 encoded)

    Args:
        config: Project configuration dict.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    prep_cfg = config["preprocessing"]
    feature_cfg = config["features"]

    numeric_features = feature_cfg["numeric"]
    binary_features = feature_cfg["binary"]
    categorical_features = feature_cfg["categorical"]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=prep_cfg["imputer_strategy"])),
    ]
    if prep_cfg.get("scaler", True):
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    drop=prep_cfg["onehot_drop"],
                    sparse_output=prep_cfg["onehot_sparse"],
                    handle_unknown=prep_cfg["onehot_handle_unknown"],
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        verbose_feature_names_out=False,
    )

    logger.info(
        "Built preprocessor: numeric=%d, categorical=%d, binary=%d",
        len(numeric_features),
        len(categorical_features),
        len(binary_features),
    )
    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    categorical_features: List[str],
    numeric_features: List[str],
    binary_features: List[str],
) -> List[str]:
    """Extract human-readable feature names from a fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        categorical_features: Original categorical column names.
        numeric_features: Original numeric column names.
        binary_features: Original binary column names.

    Returns:
        Ordered list of output feature names.
    """
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(onehot.get_feature_names_out(categorical_features))
    return numeric_features + cat_names + binary_features


def preprocess_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply binary encoding, build preprocessor, fit on train, transform both sets.

    Args:
        X_train: Raw training features (before any encoding).
        X_test: Raw test features (before any encoding).
        config: Project configuration dict.

    Returns:
        Tuple of (X_train_processed, X_test_processed, feature_names).
    """
    # Apply feature engineering before encoding/preprocessing
    X_train = build_features(X_train, config)
    X_test = build_features(X_test, config)

    binary_map = config["features"].get("binary_map", {})
    X_train = encode_binary_columns(X_train, binary_map)
    X_test = encode_binary_columns(X_test, binary_map)

    preprocessor = build_preprocessor(config)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = get_feature_names(
        preprocessor,
        config["features"]["categorical"],
        config["features"]["numeric"],
        config["features"]["binary"],
    )

    logger.info(
        "Preprocessed shapes — train: %s, test: %s, features: %d",
        X_train_arr.shape,
        X_test_arr.shape,
        len(feature_names),
    )
    return X_train_arr, X_test_arr, feature_names
