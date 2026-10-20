"""Integration tests for the FastAPI prediction service."""

import os

import joblib
import pytest
from fastapi.testclient import TestClient

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    split_features_target,
)
from src.train import MODEL_SPECS
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory, sample_raw_df=None):
    # Train a tiny model and point the API at it via MODEL_PATH.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 60
    data = {c: rng.integers(0, 4, n).astype(float) for c in config.FEATURE_COLUMNS}
    data["age"] = rng.integers(29, 77, n).astype(float)
    data["target"] = rng.integers(0, 2, n).astype(float)
    df = clean_data(pd.DataFrame(data))
    X, y = split_features_target(df)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", MODEL_SPECS["logistic_regression"]["estimator"]),
        ]
    )
    pipeline.fit(X, y)

    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    os.environ["MODEL_PATH"] = str(model_path)

    # Import the app and reset any cached model so it picks up MODEL_PATH.
    import src.api as api_module

    api_module._model = None
    with TestClient(api_module.app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client):
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("Disease", "No Disease")


def test_predict_validation_error(client):
    bad = {"age": 999, "sex": 5}  # out of range + missing fields
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "heart_predictions_total" in resp.text or "http_requests" in resp.text
