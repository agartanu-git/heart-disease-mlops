#!/usr/bin/env python3
"""Exploratory Data Analysis for the Heart Disease dataset.

Generates professional visualizations (class balance, histograms, correlation
heatmap, boxplots) and a text summary. All figures are written to
``reports/figures`` so they can be embedded in the final report.

Usage:
    python -m src.eda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / containers
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_preprocessing import get_clean_dataset  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")


def plot_class_balance(df, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    ax = sns.barplot(
        x=counts.index, y=counts.values, hue=counts.index, legend=False, palette="Set2"
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
    ax.set_ylabel("Count")
    ax.set_title("Target Class Balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "class_balance.png", dpi=120)
    plt.close()


def plot_histograms(df, out_dir: Path) -> None:
    numeric = config.NUMERIC_FEATURES
    n = len(numeric)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, col in enumerate(numeric, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(
            data=df,
            x=col,
            hue=config.TARGET_COLUMN,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(out_dir / "histograms.png", dpi=120)
    plt.close()


def plot_correlation_heatmap(df, out_dir: Path) -> None:
    plt.figure(figsize=(11, 9))
    corr = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=120)
    plt.close()


def plot_boxplots(df, out_dir: Path) -> None:
    numeric = config.NUMERIC_FEATURES
    cols = 3
    rows = (len(numeric) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, col in enumerate(numeric, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(
            data=df,
            x=config.TARGET_COLUMN,
            y=col,
            hue=config.TARGET_COLUMN,
            legend=False,
            palette="Set3",
        )
        plt.title(f"{col} by target")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplots.png", dpi=120)
    plt.close()


def write_summary(df, out_dir: Path) -> None:
    summary = {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURE_COLUMNS),
        "missing_values": {
            k: int(v) for k, v in df[config.FEATURE_COLUMNS].isna().sum().items()
        },
        "class_balance": {
            str(k): int(v)
            for k, v in df[config.TARGET_COLUMN].value_counts().sort_index().items()
        },
        "numeric_describe": json.loads(
            df[config.NUMERIC_FEATURES].describe().to_json()
        ),
    }
    with open(out_dir / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["class_balance"], indent=2))


def main() -> None:
    config.ensure_dirs()
    df = get_clean_dataset()
    out_dir = config.FIGURES_DIR

    plot_class_balance(df, out_dir)
    plot_histograms(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_boxplots(df, out_dir)
    write_summary(df, out_dir)
    print(f"EDA figures written to {out_dir}")


if __name__ == "__main__":
    main()
