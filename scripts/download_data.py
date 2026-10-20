#!/usr/bin/env python3
"""Download the UCI Heart Disease (Cleveland) dataset.

Usage:
    python scripts/download_data.py

The script downloads the raw ``processed.cleveland.data`` file, attaches the
canonical column headers and writes a clean CSV to ``data/raw/heart_disease.csv``.
It honours ``HTTP_PROXY`` / ``HTTPS_PROXY`` environment variables so it works
from behind a corporate proxy.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

# Allow running as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

# Primary and mirror URLs for the processed Cleveland dataset.
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/data.csv",
]


def download() -> str:
    """Fetch the raw dataset text from the first reachable URL."""
    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            print(f"Downloading from {url} ...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            if resp.text.strip():
                return resp.text
        except Exception as exc:  # noqa: BLE001 - report and try next mirror
            print(f"  failed: {exc}")
            last_error = exc
    raise RuntimeError(f"All download URLs failed: {last_error}")


def main() -> None:
    config.ensure_dirs()
    raw_text = download()

    # The classic Cleveland file is headerless with 14 columns.
    first_line = raw_text.strip().splitlines()[0]
    has_header = any(c.isalpha() for c in first_line)

    if has_header:
        df = pd.read_csv(io.StringIO(raw_text))
        # Normalise a possibly different target column name.
        if "num" in df.columns and config.TARGET_COLUMN not in df.columns:
            df = df.rename(columns={"num": config.TARGET_COLUMN})
        df = df[[c for c in config.COLUMN_NAMES if c in df.columns]]
    else:
        df = pd.read_csv(io.StringIO(raw_text), header=None, names=config.COLUMN_NAMES)

    out_path = config.RAW_DATA_FILE
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
