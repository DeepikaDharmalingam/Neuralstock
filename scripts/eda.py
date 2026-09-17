"""
eda.py

Generates exploratory data analysis plots from the raw NeuralStock
dataset. Produces the visuals referenced in the notebook (Section 2)
and the project report (Section 3): sales distribution, category
averages, monthly seasonality, day-of-week pattern, and an ACF plot.

Run:
    python eda.py --data data/ecommerce_inventory_demand.csv --out outputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import matplotlib
# Only force the headless Agg backend when this module is run as a script.
# Importing it from a Jupyter notebook must NOT clobber the inline backend,
# otherwise plt.show() there warns "FigureCanvasAgg is non-interactive".
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from preprocess import TARGET, handle_missing_values, load_raw

sns.set_theme(style="whitegrid")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def manual_acf(series: np.ndarray, nlags: int = 20) -> list:
    """Compute the autocorrelation function without requiring statsmodels.

    Args:
        series: 1-D array of a single product's units_sold, in chronological order.
        nlags: Number of lags to compute.

    Returns:
        List of ACF values, index 0 through nlags (index 0 is always 1.0).
    """
    series = series - series.mean()
    denom = np.sum(series ** 2)
    return [1.0] + [
        float(np.sum(series[lag:] * series[:-lag]) / denom) for lag in range(1, nlags + 1)
    ]


def run_eda(data_path: str, out_dir: str) -> None:
    """Generate and save all EDA plots for the report and notebook.

    Args:
        data_path: Path to the raw NeuralStock CSV.
        out_dir: Directory to write PNG plots into (created if missing).

    Returns:
        None. Plots are written to out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(data_path)
    df = handle_missing_values(raw_df)

    print("Rows:", len(df), "| Products:", df["product_id"].nunique())
    print("Missing values (raw) by column:")
    print(raw_df.isnull().sum())

    # 1. Distribution of units_sold
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of Daily Units Sold")
    fig.tight_layout()
    fig.savefig(out / "eda_units_sold_distribution.png", dpi=130)
    plt.close(fig)

    # 2. Category-wise average demand
    fig, ax = plt.subplots(figsize=(7, 4))
    cat_avg = df.groupby("product_category")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=cat_avg.index, y=cat_avg.values, ax=ax)
    ax.set_title("Average Units Sold by Product Category")
    fig.tight_layout()
    fig.savefig(out / "eda_category_avg_demand.png", dpi=130)
    plt.close(fig)

    # 3. Monthly seasonality
    fig, ax = plt.subplots(figsize=(7, 4))
    monthly = df.groupby("month")[TARGET].mean()
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", ax=ax)
    ax.set_title("Average Units Sold by Month (Seasonality)")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    fig.savefig(out / "eda_monthly_seasonality.png", dpi=130)
    plt.close(fig)

    # 4. Day-of-week pattern
    fig, ax = plt.subplots(figsize=(6, 4))
    dow_avg = df.groupby("day_of_week")[TARGET].mean()
    sns.barplot(x=dow_avg.index, y=dow_avg.values, ax=ax)
    ax.set_title("Average Units Sold by Day of Week (0=Mon)")
    fig.tight_layout()
    fig.savefig(out / "eda_dow_pattern.png", dpi=130)
    plt.close(fig)

    # 5. ACF for the most-observed product
    top_product = df["product_id"].value_counts().idxmax()
    series = df[df["product_id"] == top_product].sort_values("date")[TARGET].reset_index(drop=True)
    acf_vals = manual_acf(series.to_numpy(), nlags=20)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(range(len(acf_vals)), acf_vals)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Autocorrelation (ACF) of units_sold — {top_product}")
    ax.set_xlabel("lag (observation index, not calendar days)")
    fig.tight_layout()
    fig.savefig(out / f"eda_acf_{top_product}.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved 5 EDA plots to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NeuralStock EDA plots")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "ecommerce_inventory_demand.csv"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()
    run_eda(args.data, args.out)
