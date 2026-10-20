#!/usr/bin/env python3
"""Train, tune and track Heart Disease classification models.

Trains at least two models (Logistic Regression and Random Forest) inside a
full preprocessing + estimator ``Pipeline``. Hyper-parameters are tuned with
``GridSearchCV`` using stratified cross-validation. Every run is logged to
MLflow (params, metrics, ROC curve, confusion matrix). The best model by
ROC-AUC is persisted to ``models/model.joblib`` for serving.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# MLflow uses GitPython for run provenance; silence its warning when git is
# unavailable in the execution environment (e.g. minimal containers).
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    get_clean_dataset,
    split_features_target,
)

# --- Model search space ---------------------------------------------------
MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=config.RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_split": [2, 5],
        },
    },
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _log_plots(pipeline, X_test, y_test, name: str, out_dir: Path) -> None:
    # Confusion matrix
    cm_path = out_dir / f"{name}_confusion_matrix.png"
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # ROC curve
    roc_path = out_dir / f"{name}_roc_curve.png"
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=120)
    plt.close()

    mlflow.log_artifact(str(cm_path), artifact_path="plots")
    mlflow.log_artifact(str(roc_path), artifact_path="plots")


def train() -> dict:
    config.ensure_dirs()

    df = get_clean_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Persist the train/test split for reproducibility.
    pd.concat([X_train, y_train], axis=1).to_csv(
        config.PROCESSED_TRAIN_FILE, index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(config.PROCESSED_TEST_FILE, index=False)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    results = {}
    best_overall = {"name": None, "roc_auc": -1.0, "pipeline": None, "metrics": None}

    for name, spec in MODEL_SPECS.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("model", spec["estimator"]),
                ]
            )
            search = GridSearchCV(
                pipeline,
                param_grid=spec["param_grid"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_

            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)
            cv_best = float(search.best_score_)

            # --- MLflow logging ---
            mlflow.log_param("model_type", name)
            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("cv_roc_auc", cv_best)
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            _log_plots(best_pipeline, X_test, y_test, name, config.FIGURES_DIR)
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            print(f"[{name}] best_params={search.best_params_}")
            print(f"[{name}] cv_roc_auc={cv_best:.4f} test={metrics}")

            results[name] = {
                "best_params": search.best_params_,
                "cv_roc_auc": cv_best,
                "test_metrics": metrics,
            }

            if metrics["roc_auc"] > best_overall["roc_auc"]:
                best_overall.update(
                    name=name,
                    roc_auc=metrics["roc_auc"],
                    pipeline=best_pipeline,
                    metrics=metrics,
                )

    # Persist the winning model + metrics for serving.
    joblib.dump(best_overall["pipeline"], config.MODEL_FILE)
    summary = {
        "best_model": best_overall["name"],
        "best_metrics": best_overall["metrics"],
        "all_results": results,
    }
    with open(config.METRICS_FILE, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(
        f"\nBest model: {best_overall['name']} "
        f"(ROC-AUC={best_overall['roc_auc']:.4f}) -> {config.MODEL_FILE}"
    )
    return summary


if __name__ == "__main__":
    train()
