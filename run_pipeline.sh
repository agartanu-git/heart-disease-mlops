#!/usr/bin/env bash
# One-shot local pipeline: download data -> EDA -> train -> report metrics.
# Assumes the virtual environment is already activated.
set -euo pipefail

echo "==> [1/3] Downloading dataset"
python scripts/download_data.py

echo "==> [2/3] Running EDA"
python -m src.eda

echo "==> [3/3] Training & tracking models"
python -m src.train

echo "==> Done. Model at models/model.joblib"
echo "    View experiments: mlflow ui --backend-store-uri ./mlruns"
echo "    Serve API:        uvicorn src.api:app --host 0.0.0.0 --port 8000"
