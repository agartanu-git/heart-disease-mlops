"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small raw-like dataframe including a '?' missing marker and the
    original multi-class (0-4) target to exercise cleaning logic."""
    rng = np.random.default_rng(0)
    n = 40
    data = {
        "age": rng.integers(29, 77, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(94, 200, n).astype(float),
        "chol": rng.integers(126, 410, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(71, 202, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": rng.integers(0, 5, n).astype(float),  # 0-4 severity
    }
    df = pd.DataFrame(data)
    # Inject a couple of missing values.
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    return df
