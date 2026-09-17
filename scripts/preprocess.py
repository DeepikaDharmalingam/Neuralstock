"""
preprocess.py

Data cleaning and feature engineering for the NeuralStock demand
forecasting pipeline. All functions are pure (take a DataFrame, return a
DataFrame) so they can be unit-tested and reused identically inside the
notebook, train.py, and the Streamlit app's inference path.

Design notes:
    - Missing values in `units_sold` are forward-filled *within each
      product_id's own chronological series* (never across products).
    - The MinMaxScaler must be fit on the training split ONLY. This module
      exposes `fit_scaler` / `apply_scaler` as separate steps specifically
      to prevent accidental data leakage.
    - Lag and rolling-window features are computed per product_id, sorted
      by date, so a lag-7 value never crosses into a different SKU.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = [
    "unit_price",
    "stock_on_hand",
    "reorder_point",
    "discount_pct",
    "supplier_lead_days",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_30",
    "roll_std_30",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]
TARGET = "units_sold"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw NeuralStock CSV and parse dates.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A DataFrame with `date` parsed as a datetime64 column and rows
        sorted by product_id then date (required for correct lag/rolling
        feature computation downstream).
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing `units_sold` values via per-product forward-fill.

    Any remaining leading NaNs (a product's very first observation was
    missing, so there is nothing to forward-fill from) are back-filled as
    a last resort, then any still-missing rows are dropped.

    Args:
        df: DataFrame sorted by product_id, date.

    Returns:
        DataFrame with no missing values in the target column.
    """
    df = df.copy()
    df[TARGET] = df.groupby("product_id")[TARGET].transform(
        lambda s: s.ffill().bfill()
    )
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def remove_price_outliers(df: pd.DataFrame, column: str = "unit_price") -> pd.DataFrame:
    """Clip outliers in a numeric column using the IQR method, per category.

    Args:
        df: Input DataFrame.
        column: Column to clip outliers on.

    Returns:
        DataFrame with the column's extreme values clipped to
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR], computed within each product_category
        (prices vary hugely by category, so a global IQR would over-clip).
    """
    df = df.copy()

    def _clip(group: pd.Series) -> pd.Series:
        q1, q3 = group.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return group.clip(lower, upper)

    df[column] = df.groupby("product_category")[column].transform(_clip)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekend flag and cyclical (sin/cos) encodings for month and day-of-week.

    Args:
        df: DataFrame containing `day_of_week` (0-6) and `month` (1-12).

    Returns:
        DataFrame with added columns: is_weekend, month_sin, month_cos,
        dow_sin, dow_cos.
    """
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-7, lag-14, and 7/30-period rolling mean & std of units_sold.

    All windows are computed per product_id, in chronological order, and
    shifted by one observation so that the current row's target is never
    used to compute its own features (no leakage from the label itself).

    Args:
        df: DataFrame sorted by product_id, date, with a `units_sold` column.

    Returns:
        DataFrame with added columns: lag_7, lag_14, roll_mean_7,
        roll_std_7, roll_mean_30, roll_std_30. Rows without enough history
        for a given SKU will contain NaNs in those columns.
    """
    df = df.copy()
    grouped = df.groupby("product_id")[TARGET]

    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)
    df["roll_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_std_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).std())
    df["roll_mean_30"] = grouped.transform(lambda s: s.shift(1).rolling(30).mean())
    df["roll_std_30"] = grouped.transform(lambda s: s.shift(1).rolling(30).std())

    # Keep an UNSCALED copy of roll_mean_7. `roll_mean_7` itself is one of
    # NUMERIC_FEATURES and gets MinMax-scaled to [0, 1] by apply_scaler, so
    # it can no longer serve as a real-units baseline once scaled. This copy
    # is deliberately excluded from NUMERIC_FEATURES so apply_scaler leaves
    # it alone — the LSTM target-residual trick in model.py needs the
    # original units_sold-scale value to add back onto its predictions.
    df["roll_mean_7_raw"] = df["roll_mean_7"]
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `product_category`.

    Args:
        df: DataFrame containing a `product_category` column.

    Returns:
        DataFrame with `product_category` replaced by one-hot indicator
        columns (category_Electronics, category_Apparel, ...).
    """
    return pd.get_dummies(df, columns=["product_category"], prefix="category")


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test strictly by date, per product, with no shuffling.

    Splitting per-product (rather than globally by an overall date cutoff)
    keeps every SKU represented in both splits even though products were
    sampled on different days.

    Args:
        df: DataFrame sorted by product_id, date.
        test_size: Fraction of each product's most recent rows held out.

    Returns:
        (train_df, test_df) tuple, each still sorted by product_id, date.
    """

    def _split(group: pd.DataFrame) -> pd.Series:
        n_test = max(1, int(len(group) * test_size))
        labels = pd.Series("train", index=group.index)
        labels.iloc[-n_test:] = "test"
        return labels

    split_labels = df.groupby("product_id", group_keys=False).apply(_split, include_groups=False)
    train_df = df[split_labels == "train"].reset_index(drop=True)
    test_df = df[split_labels == "test"].reset_index(drop=True)
    return train_df, test_df


def bridge_train_test_for_windows(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sequence_length: int = 14
) -> pd.DataFrame:
    """Prepend each product's trailing training rows onto its test split.

    SlidingWindowDataset builds one fixed-length window of history ending
    immediately before each target row. Without this bridge, the first
    `sequence_length` rows of every product's test split have no prior
    test-set history to draw a window from and are silently dropped —
    on this dataset that discards roughly 75% of the test targets and
    makes the LSTM's test set incomparable to the MLP's (which uses every
    row). Prepending the tail of TRAINING data (never future/test data)
    is not leakage: it only gives the model the same history a forecaster
    would already have on day 1 of the test period.

    Args:
        train_df: Training split, sorted by product_id, date. Pass the
            SCALED frame (post apply_scaler) so its feature columns match
            what the test frame will be windowed on; unscaled columns such
            as units_sold and roll_mean_7_raw pass through unchanged.
        test_df: Test split, same columns and scaling as train_df.
        sequence_length: Window length that will be used by
            SlidingWindowDataset; must match the value passed there.

    Returns:
        One DataFrame per product of [last `sequence_length` train rows]
        followed by [all test rows], concatenated and sorted by
        product_id, date. Windowing this with SlidingWindowDataset at the
        same sequence_length yields exactly one window per original test
        row (fewer only for a product with less than sequence_length rows
        of training history available).
    """
    frames = []
    for pid, test_group in test_df.groupby("product_id", sort=False):
        history = train_df[train_df["product_id"] == pid].tail(sequence_length)
        frames.append(pd.concat([history, test_group], ignore_index=True))
    if not frames:
        return test_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def fit_scaler(train_df: pd.DataFrame, columns: list = None) -> MinMaxScaler:
    """Fit a MinMaxScaler on the TRAINING split only.

    Args:
        train_df: Training DataFrame (post feature engineering).
        columns: Numeric columns to scale. Defaults to NUMERIC_FEATURES.

    Returns:
        A fitted MinMaxScaler instance. Save this with pickle/joblib
        (as scaler.pkl) and reuse it for both the test set and inference —
        never re-fit on test or production data.
    """
    columns = columns or NUMERIC_FEATURES
    scaler = MinMaxScaler()
    scaler.fit(train_df[columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, columns: list = None) -> pd.DataFrame:
    """Transform a DataFrame's numeric columns with an already-fitted scaler.

    Args:
        df: DataFrame to transform (train, test, or new inference data).
        scaler: A MinMaxScaler previously fit on the training split.
        columns: Numeric columns to scale. Defaults to NUMERIC_FEATURES.

    Returns:
        A copy of df with the given columns replaced by their scaled values.
    """
    columns = columns or NUMERIC_FEATURES
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


def run_pipeline(raw_csv_path: str) -> dict:
    """Run the full cleaning + feature engineering pipeline end-to-end.

    Args:
        raw_csv_path: Path to the raw NeuralStock CSV.

    Returns:
        A dict with keys: train_df, test_df, scaler, feature_columns —
        ready to feed into the PyTorch Dataset/DataLoader in model.py.
    """
    df = load_raw(raw_csv_path)
    df = handle_missing_values(df)
    df = remove_price_outliers(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)

    # Drop rows that don't yet have enough history for the longest window
    # (roll_mean_30 needs 30 prior observations for that SKU).
    df = df.dropna(subset=["roll_mean_30", "roll_std_30", "lag_14"]).reset_index(drop=True)

    train_df, test_df = chronological_split(df, test_size=0.2)
    scaler = fit_scaler(train_df)
    train_scaled = apply_scaler(train_df, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    return {
        "train_df": train_scaled,
        "test_df": test_scaled,
        "scaler": scaler,
        "feature_columns": NUMERIC_FEATURES,
        "raw_train_df": train_df,
        "raw_test_df": test_df,
    }


if __name__ == "__main__":
    result = run_pipeline(str(PROJECT_ROOT / "data" / "ecommerce_inventory_demand.csv"))
    print("Train shape:", result["train_df"].shape)
    print("Test shape:", result["test_df"].shape)
    print("Feature columns:", result["feature_columns"])
