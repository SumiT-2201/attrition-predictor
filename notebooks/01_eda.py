"""Exploratory Data Analysis — Employee Attrition Predictor.

Runs the full EDA pipeline headlessly (no display required):
  1. Load and validate raw data.
  2. Generate five EDA figures saved to reports/figures/.
  3. Split into train/test, fit the preprocessing pipeline, and save
     processed CSVs to data/processed/.

Run from the project root:
    python notebooks/01_eda.py
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import.

import sys
from pathlib import Path

# Allow imports from project root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    PIPELINE_FILE,
    RANDOM_STATE,
)
from src.data import load_raw_data, summarise, validate_schema
from src.preprocess import (
    build_preprocessing_pipeline,
    drop_leakage_columns,
    encode_binary_columns,
    save_pipeline,
    split_features_target,
    train_test_split_stratified,
)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def save_fig(name: str) -> None:
    """Save the current matplotlib figure to FIGURES_DIR.

    Args:
        name: Output filename (e.g. ``"class_distribution.png"``).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of Attrition value counts with count labels.

    Args:
        df: DataFrame with an integer ``Attrition`` column (0/1).
    """
    counts = df["Attrition"].value_counts().sort_index()
    labels = ["No (Retained)", "Yes (Attrition)"]
    total = counts.sum()
    pct = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="black")

    # Add count labels above each bar.
    for bar, cnt, p in zip(bars, counts.values, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{cnt}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(
        f"Attrition Distribution  "
        f"(Yes: {pct.iloc[1]}%  /  No: {pct.iloc[0]}%)",
        fontsize=12,
    )
    ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig("class_distribution.png")
    print("Saved class_distribution.png")


def plot_age_histogram(df: pd.DataFrame) -> None:
    """Overlapping histograms of Age split by Attrition class.

    Args:
        df: DataFrame with ``Age`` (int) and ``Attrition`` (0/1) columns.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, colour, name in [
        (0, "#2ecc71", "Retained"),
        (1, "#e74c3c", "Attrition"),
    ]:
        subset = df.loc[df["Attrition"] == label, "Age"]
        ax.hist(subset, bins=20, alpha=0.6, color=colour, label=name, edgecolor="white")

    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Attrition")
    ax.legend()
    plt.tight_layout()
    save_fig("age_histogram.png")
    print("Saved age_histogram.png")


def plot_monthly_income_histogram(df: pd.DataFrame) -> None:
    """Overlapping histograms of MonthlyIncome split by Attrition class.

    Args:
        df: DataFrame with ``MonthlyIncome`` (int) and ``Attrition`` (0/1).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, colour, name in [
        (0, "#2ecc71", "Retained"),
        (1, "#e74c3c", "Attrition"),
    ]:
        subset = df.loc[df["Attrition"] == label, "MonthlyIncome"]
        ax.hist(subset, bins=25, alpha=0.6, color=colour, label=name, edgecolor="white")

    ax.set_xlabel("Monthly Income ($)")
    ax.set_ylabel("Count")
    ax.set_title("Monthly Income Distribution by Attrition")
    ax.legend()
    plt.tight_layout()
    save_fig("monthly_income_histogram.png")
    print("Saved monthly_income_histogram.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Seaborn heatmap of the numeric correlation matrix.

    Args:
        df: DataFrame containing numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax, linewidths=0.3)
    ax.set_title("Numeric Feature Correlation Matrix")
    plt.tight_layout()
    save_fig("correlation_heatmap.png")
    print("Saved correlation_heatmap.png")


def plot_categoricals_vs_attrition(df: pd.DataFrame) -> None:
    """2×2 grid of grouped bar charts for key categorical features vs attrition.

    Args:
        df: DataFrame with ``Attrition`` (0/1) and categorical columns.
    """
    features = ["BusinessTravel", "Department", "JobRole", "MaritalStatus"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, feat in zip(axes.flatten(), features):
        # Mean attrition rate per category.
        rates = df.groupby(feat)["Attrition"].mean().sort_values(ascending=False)
        ax.bar(rates.index, rates.values, color="#3498db", edgecolor="black")
        ax.set_title(f"{feat} vs Mean Attrition Rate")
        ax.set_ylabel("Mean Attrition Rate")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Categorical Features vs Attrition Rate", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig("top_categoricals.png")
    print("Saved top_categoricals.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step A: Load and validate
    # ------------------------------------------------------------------
    print("Loading raw data…")
    raw = load_raw_data()
    validate_schema(raw)
    summarise(raw)

    df = drop_leakage_columns(raw)
    df = encode_binary_columns(df)
    print(f"\nShape after cleaning: {df.shape}")

    # ------------------------------------------------------------------
    # Step B: EDA figures
    # ------------------------------------------------------------------
    print("\nGenerating EDA figures…")
    plot_class_distribution(df)
    plot_age_histogram(df)
    plot_monthly_income_histogram(df)
    plot_correlation_heatmap(df)
    plot_categoricals_vs_attrition(df)
    print("All figures saved.")

    # ------------------------------------------------------------------
    # Step C: Split, preprocess, and save processed CSVs
    # ------------------------------------------------------------------
    print("\nSplitting into train / test sets…")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("Building and fitting preprocessing pipeline…")
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feat_names = preprocessor.get_feature_names_out().tolist()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Reconstruct DataFrames with feature names, then add target column.
    train_df = pd.DataFrame(X_train_proc, columns=feat_names)
    train_df["Attrition"] = y_train.values
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    print(f"Saved train.csv  ({train_df.shape})")

    test_df = pd.DataFrame(X_test_proc, columns=feat_names)
    test_df["Attrition"] = y_test.values
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)
    print(f"Saved test.csv   ({test_df.shape})")

    save_pipeline(preprocessor)
    print(f"Pipeline saved → {PIPELINE_FILE}")
    print("\nEDA complete.")
