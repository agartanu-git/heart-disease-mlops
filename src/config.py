"""Central configuration for the Heart Disease MLOps project.

All paths are resolved relative to the project root so scripts work from a
clean checkout regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "heart_disease.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

# --- Dataset definition ---------------------------------------------------
# UCI Heart Disease (processed Cleveland). The raw .data file has no header;
# these are the canonical column names.
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Original target is 0-4 (severity). We convert to binary: 0 = no disease,
# 1 = disease present.
TARGET_COLUMN = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Reproducibility ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- MLflow ---------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = f"file://{(PROJECT_ROOT / 'mlruns').as_posix()}"


def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
