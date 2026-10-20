"""Data loading, cleaning and preprocessing for the Heart Disease dataset.

This module is the single source of truth for how raw patient records are
turned into model-ready features. The same code path is used in training and
at inference time (via the fitted ``ColumnTransformer``) to guarantee full
reproducibility.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw UCI heart disease CSV.

    The raw file may use ``?`` to mark missing values; these are coerced to
    ``NaN`` so downstream imputation can handle them.
    """
    path = path or config.RAW_DATA_FILE
    df = pd.read_csv(path, na_values=["?"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw records.

    Steps:
    * Drop exact duplicate rows.
    * Convert the multi-class target (0-4 severity) into a binary label
      (0 = no disease, 1 = disease present).
    * Ensure numeric dtypes for all feature columns.
    """
    df = df.copy()
    df = df.drop_duplicates()

    # Binarise the target: any value > 0 means disease is present.
    if config.TARGET_COLUMN in df.columns:
        df[config.TARGET_COLUMN] = (
            pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce") > 0
        ).astype(int)

    # Coerce every feature to numeric; ``?`` -> NaN already handled on load.
    for col in config.FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no target.
    df = df.dropna(subset=[config.TARGET_COLUMN])
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline.

    Numeric features are median-imputed then standard-scaled; categorical
    features are most-frequent-imputed then one-hot encoded. Returning an
    unfitted transformer lets callers fit it only on training data, which
    prevents leakage.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_clean_dataset(path=None) -> pd.DataFrame:
    """Convenience helper returning a fully cleaned dataframe."""
    return clean_data(load_raw_data(path))
