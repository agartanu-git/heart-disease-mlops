"""FastAPI service exposing the Heart Disease classifier.

Endpoints:
* ``GET  /``         - service info
* ``GET  /health``   - liveness/readiness probe
* ``POST /predict``  - single prediction (JSON in, prediction + confidence out)
* ``POST /predict/batch`` - batch prediction
* ``GET  /metrics``  - Prometheus metrics (added by the instrumentator)

Structured request/response logging and a custom Prometheus counter/histogram
provide the observability required for monitoring.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src import config
from src.schemas import (
    BatchRequest,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)

# --- Logging --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")


# --- Custom Prometheus metrics -------------------------------------------
# Guard against duplicate registration (e.g. when the module is reimported in
# tests) by reusing the already-registered collector if present.
def _get_or_create(metric_cls, name, documentation, labelnames=()):
    try:
        return metric_cls(name, documentation, labelnames)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


PREDICTION_COUNTER = _get_or_create(
    Counter,
    "heart_predictions_total",
    "Total predictions served, labelled by predicted class",
    ["predicted_class"],
)
PREDICTION_LATENCY = _get_or_create(
    Histogram,
    "heart_prediction_latency_seconds",
    "Latency of the /predict handler in seconds",
)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
)

# Expose Prometheus metrics at /metrics. Must be configured before the app
# starts serving, so it lives at module scope rather than in a startup hook.
Instrumentator().instrument(app).expose(app, include_in_schema=False)

_model = None


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(config.MODEL_FILE)))


def load_model():
    """Load the serialized pipeline lazily and cache it."""
    global _model
    if _model is None:
        path = _model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `python -m src.train` first."
            )
        _model = joblib.load(path)
        logger.info("Model loaded from %s", path)
    return _model


@app.on_event("startup")
def _startup() -> None:
    try:
        load_model()
    except FileNotFoundError as exc:
        logger.warning("Startup without model: %s", exc)


def _features_to_frame(features: PatientFeatures) -> pd.DataFrame:
    data = features.model_dump()
    return pd.DataFrame([data])[config.FEATURE_COLUMNS]


def _predict_frame(df: pd.DataFrame) -> list[PredictionResponse]:
    model = load_model()
    preds = model.predict(df)
    probas = model.predict_proba(df)
    responses = []
    for pred, proba in zip(preds, probas):
        pred = int(pred)
        prob_disease = float(proba[1])
        confidence = float(max(proba))
        PREDICTION_COUNTER.labels(predicted_class=str(pred)).inc()
        responses.append(
            PredictionResponse(
                prediction=pred,
                label="Disease" if pred == 1 else "No Disease",
                confidence=round(confidence, 4),
                probability_disease=round(prob_disease, 4),
            )
        )
    return responses


@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/predict/batch", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        model = load_model()
        model_type = (
            type(model.named_steps["model"]).__name__
            if hasattr(model, "named_steps")
            else type(model).__name__
        )
        return HealthResponse(status="ok", model_loaded=True, model_type=model_type)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    start = time.time()
    try:
        df = _features_to_frame(features)
        result = _predict_frame(df)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        elapsed = time.time() - start
        PREDICTION_LATENCY.observe(elapsed)
    logger.info(
        "predict input=%s -> prediction=%s confidence=%.4f (%.1f ms)",
        features.model_dump(),
        result.prediction,
        result.confidence,
        elapsed * 1000,
    )
    return result


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchRequest) -> list[PredictionResponse]:
    try:
        df = pd.DataFrame([r.model_dump() for r in request.records])[
            config.FEATURE_COLUMNS
        ]
        results = _predict_frame(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info("batch predict n=%d", len(results))
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
