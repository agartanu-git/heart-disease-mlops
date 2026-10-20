"""Unit tests for model training logic and the serving pipeline."""

import numpy as np
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor, split_features_target
from src.train import MODEL_SPECS, evaluate


def test_evaluate_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])
    metrics = evaluate(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_model_specs_contains_two_models():
    assert "logistic_regression" in MODEL_SPECS
    assert "random_forest" in MODEL_SPECS
    assert len(MODEL_SPECS) >= 2


def test_pipeline_fits_and_predicts(sample_raw_df):
    from src.data_preprocessing import clean_data

    cleaned = clean_data(sample_raw_df)
    X, y = split_features_target(cleaned)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", MODEL_SPECS["logistic_regression"]["estimator"]),
        ]
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    assert len(preds) == len(y)
    assert proba.shape == (len(y), 2)
    assert set(np.unique(preds)).issubset({0, 1})
