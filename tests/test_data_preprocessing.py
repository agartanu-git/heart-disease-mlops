"""Unit tests for data cleaning and preprocessing."""

import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    split_features_target,
)


def test_clean_data_binarises_target(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert set(cleaned[config.TARGET_COLUMN].unique()).issubset({0, 1})


def test_clean_data_drops_duplicates():
    row = {c: 1.0 for c in config.COLUMN_NAMES}
    df = pd.DataFrame([row, row, row])
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_clean_data_handles_question_mark_missing():
    row = {c: 1.0 for c in config.COLUMN_NAMES}
    df = pd.DataFrame([row, row])
    df.loc[0, "ca"] = "?"
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    cleaned = clean_data(df)
    # Target still valid, ca may be NaN but imputed later, not dropped here.
    assert config.TARGET_COLUMN in cleaned.columns


def test_split_features_target_shapes(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X, y = split_features_target(cleaned)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert len(X) == len(y)
    assert y.dtype == int or y.dtype == np.int64


def test_preprocessor_transforms_without_nan(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X, y = split_features_target(cleaned)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X, y)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
    # Row count preserved.
    assert arr.shape[0] == len(X)


def test_preprocessor_handles_unknown_category(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X, y = split_features_target(cleaned)
    pre = build_preprocessor()
    pre.fit(X, y)
    # Introduce an unseen category value for 'thal'.
    X_new = X.iloc[[0]].copy()
    X_new.loc[:, "thal"] = 99.0
    out = pre.transform(X_new)
    arr = out.toarray() if hasattr(out, "toarray") else out
    assert arr.shape[0] == 1
