"""preprocess.py

Cleaning and feature engineering for the NeuralStock demand-forecasting
pipeline. Every function is pure (DataFrame in, DataFrame out) so the
notebook, ``train.py`` and the Streamlit inference path all share exactly
one implementation.

Design rules enforced here
--------------------------
1. ``units_sold`` is forward-filled *within each SKU's own chronological
   series*, never across SKUs.
2. Lag / rolling windows are computed per ``product_id`` on a **complete
   daily calendar**, so ``lag_7`` really is "seven days ago". If the input
   frame has gaps, :func:`reindex_daily` fills them first.
3. Every window is ``.shift(1)``-ed before aggregating, so a row's own
   target never contributes to its own features.
4. The MinMaxScaler is fit on the TRAIN split only - ``fit_scaler`` and
   ``apply_scaler`` are deliberately separate calls to make leakage hard.
5. The split is chronological with a three-way TRAIN / VALIDATION / TEST
   cut. Validation drives early stopping and LR scheduling; the test split
   is touched exactly once, at final scoring.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET = "units_sold"

#: Numeric columns scaled to [0, 1] by the MinMaxScaler and fed to the models.
NUMERIC_FEATURES = [
    "prev_demand",      # yesterday's actual sales (lag-1)
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_30",
    "roll_std_30",
    "unit_price",
    "stock_on_hand",
    "reorder_point",
    "discount_pct",
    "supplier_lead_days",
    "is_promotion",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "doy_sin",
    "doy_cos",
]

#: Columns that must survive scaling untouched (targets / baselines / keys).
PASSTHROUGH_COLUMNS = ["date", "product_id", "product_category", TARGET,
                       "roll_mean_7_raw"]


def load_raw(path: str) -> pd.DataFrame:
    """Load a NeuralStock CSV and parse dates.

    Args:
        path: Path to the CSV file.

    Returns:
        A DataFrame with ``date`` as datetime64, sorted by product_id then
        date (required for correct lag / rolling computation downstream).
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["product_id", "date"]).reset_index(drop=True)


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex every SKU onto a gap-free daily calendar.

    Time-series lags are only meaningful on a regular grid. If a SKU is
    missing calendar days, this inserts those rows with ``units_sold`` set
    to NaN (imputed by :func:`handle_missing_values`) and static attributes
    carried forward, so ``lag_7`` is always exactly seven days back.

    Args:
        df: DataFrame sorted by product_id, date.

    Returns:
        A DataFrame covering every calendar day between each SKU's first and
        last observation. Returns the input unchanged if it is already dense.
    """
    frames = []
    static = ["product_category", "reorder_point", "supplier_lead_days"]
    carry = ["unit_price", "stock_on_hand"]

    for pid, group in df.groupby("product_id", sort=False):
        group = group.set_index("date").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(full_index)
        group.index.name = "date"
        group["product_id"] = pid
        for col in static:
            if col in group:
                group[col] = group[col].ffill().bfill()
        for col in carry:
            if col in group:
                group[col] = group[col].ffill().bfill()
        for col in ("is_promotion", "discount_pct"):
            if col in group:
                group[col] = group[col].fillna(0)
        frames.append(group.reset_index())

    out = pd.concat(frames, ignore_index=True)
    out["day_of_week"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    return out.sort_values(["product_id", "date"]).reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing ``units_sold`` via per-SKU forward-fill.

    Leading NaNs (a SKU's first observation was missing, so there is nothing
    to carry forward) are back-filled as a fallback; anything still missing
    is dropped.

    Args:
        df: DataFrame sorted by product_id, date.

    Returns:
        A DataFrame with no missing values in the target column.
    """
    df = df.copy()
    df[TARGET] = df.groupby("product_id")[TARGET].transform(lambda s: s.ffill().bfill())
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def remove_price_outliers(df: pd.DataFrame, column: str = "unit_price") -> pd.DataFrame:
    """Clip outliers in a numeric column using the IQR rule, within category.

    Prices differ by orders of magnitude across categories, so a single
    global IQR would clip every Electronics SKU. Bounds are computed per
    ``product_category`` instead.

    Args:
        df: Input DataFrame.
        column: Numeric column to clip.

    Returns:
        A copy of df with the column clipped to
        ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` within each category.
    """
    df = df.copy()

    def _clip(group: pd.Series) -> pd.Series:
        q1, q3 = group.quantile([0.25, 0.75])
        iqr = q3 - q1
        return group.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    df[column] = df.groupby("product_category")[column].transform(_clip)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the weekend flag and cyclical encodings of the calendar.

    Cyclical (sin/cos) encoding keeps December adjacent to January and Sunday
    adjacent to Monday, which a plain integer column cannot express.

    Args:
        df: DataFrame containing ``date``, ``day_of_week`` (0-6) and
            ``month`` (1-12).

    Returns:
        A copy of df with is_weekend, dow_sin/cos, month_sin/cos,
        quarter_sin/cos and doy_sin/cos added.
    """
    df = df.copy()
    day_of_year = df["date"].dt.dayofyear
    quarter = df["date"].dt.quarter

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-window features of ``units_sold``, per SKU.

    ``prev_demand`` (lag-1) is included alongside the lag-7 / lag-14 features
    the brief asks for. Without it neither model can see yesterday's actual
    sales, which is the single strongest predictor of today's - in the
    original pipeline this was the main reason the LSTM barely beat the naive
    baseline.

    Args:
        df: DataFrame sorted by product_id, date, with a ``units_sold`` column
            on a gap-free daily calendar.

    Returns:
        A copy of df with prev_demand, lag_7, lag_14, roll_mean_7, roll_std_7,
        roll_mean_30, roll_std_30 and an unscaled roll_mean_7_raw added. Rows
        without enough history contain NaN in these columns.
    """
    df = df.copy()
    grouped = df.groupby("product_id")[TARGET]

    df["prev_demand"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)
    df["roll_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_std_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).std())
    df["roll_mean_30"] = grouped.transform(lambda s: s.shift(1).rolling(30).mean())
    df["roll_std_30"] = grouped.transform(lambda s: s.shift(1).rolling(30).std())

    # Unscaled copy of roll_mean_7. roll_mean_7 itself is MinMax-scaled to
    # [0, 1] by apply_scaler, so it can no longer act as a real-units
    # baseline. This copy is excluded from NUMERIC_FEATURES on purpose: the
    # naive comparator and the LSTM's residual target both need real units.
    df["roll_mean_7_raw"] = df["roll_mean_7"]
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``product_category``.

    Args:
        df: DataFrame containing a ``product_category`` column.

    Returns:
        A DataFrame with ``category_<Name>`` 0/1 indicator columns added. The
        original ``product_category`` label column is retained for reporting
        and weekly aggregation (it is never passed to a model).
    """
    out = pd.get_dummies(df, columns=["product_category"], prefix="category")
    for col in out.columns:
        if col.startswith("category_"):
            out[col] = out[col].astype(float)
    # Keep the original label too: it is not a model input, but every
    # evaluation in this project aggregates by category, so it has to
    # survive to the scoring step.
    out["product_category"] = df["product_category"].to_numpy()
    return out


def category_columns(df: pd.DataFrame) -> List[str]:
    """Return the one-hot category column names present in a frame.

    Args:
        df: A frame that has been through :func:`encode_categoricals`.

    Returns:
        A sorted list of ``category_*`` column names.
    """
    return sorted(c for c in df.columns if c.startswith("category_"))


def chronological_split(
    df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each SKU's series chronologically into train / val / test.

    No shuffling anywhere: within each SKU the oldest rows are train, the
    next block is validation, the most recent block is test. Splitting per
    SKU (rather than on one global date cut) keeps every SKU represented in
    all three splits.

    The validation split is what distinguishes this from the original
    pipeline, which selected the best checkpoint on the *test* set - that
    leaks test information into model selection and inflates the reported
    score.

    Args:
        df: DataFrame sorted by product_id, date.
        val_size: Fraction of each SKU's rows held out for validation.
        test_size: Fraction of each SKU's most recent rows held out for test.

    Returns:
        A ``(train_df, val_df, test_df)`` tuple, each sorted by
        product_id, date.
    """
    trains, vals, tests = [], [], []
    for _, group in df.groupby("product_id", sort=False):
        group = group.sort_values("date")
        n = len(group)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        trains.append(group.iloc[: n - n_val - n_test])
        vals.append(group.iloc[n - n_val - n_test: n - n_test])
        tests.append(group.iloc[n - n_test:])

    def _concat(frames):
        return pd.concat(frames, ignore_index=True).sort_values(
            ["product_id", "date"]).reset_index(drop=True)

    return _concat(trains), _concat(vals), _concat(tests)


def bridge_for_windows(history_df: pd.DataFrame, target_df: pd.DataFrame,
                       sequence_length: int = 14) -> pd.DataFrame:
    """Prepend each SKU's trailing history rows onto the split being scored.

    ``SlidingWindowDataset`` needs ``sequence_length`` prior rows before it can
    emit a window. Without this bridge the first 14 rows of every SKU's test
    split produce no sample at all, so the LSTM would be scored on a smaller,
    different sample than the MLP.

    This is not leakage: it supplies only rows that *precede* the evaluation
    period, which is exactly the history a forecaster already holds on day one
    of that period.

    Args:
        history_df: The earlier split (e.g. train), already scaled.
        target_df: The split being windowed (e.g. val or test), same columns
            and same scaling as ``history_df``.
        sequence_length: Window length; must match ``SlidingWindowDataset``.

    Returns:
        A DataFrame of ``[last sequence_length history rows] + [all target
        rows]`` per SKU, yielding exactly one window per original target row.
    """
    frames = []
    for pid, target_group in target_df.groupby("product_id", sort=False):
        history = history_df[history_df["product_id"] == pid].tail(sequence_length)
        frames.append(pd.concat([history, target_group], ignore_index=True))
    if not frames:
        return target_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def fit_scaler(train_df: pd.DataFrame, columns: List[str] = None) -> MinMaxScaler:
    """Fit a MinMaxScaler on the TRAIN split only.

    Args:
        train_df: Training DataFrame, post feature engineering.
        columns: Columns to scale. Defaults to :data:`NUMERIC_FEATURES`.

    Returns:
        A fitted MinMaxScaler. Persist it as ``scaler.pkl`` and reuse it for
        validation, test and production inference - never re-fit downstream.
    """
    columns = columns or NUMERIC_FEATURES
    scaler = MinMaxScaler()
    scaler.fit(train_df[columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler,
                 columns: List[str] = None) -> pd.DataFrame:
    """Transform numeric columns with an already-fitted scaler.

    Args:
        df: DataFrame to transform (train, val, test or new data).
        scaler: A MinMaxScaler previously fit on the train split.
        columns: Columns to scale. Defaults to :data:`NUMERIC_FEATURES`.

    Returns:
        A copy of df with those columns replaced by their scaled values. All
        columns in :data:`PASSTHROUGH_COLUMNS` are left untouched.
    """
    columns = columns or NUMERIC_FEATURES
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


def aggregate_weekly(df: pd.DataFrame, actual_col: str = "y_true",
                     pred_col: str = "y_pred") -> pd.DataFrame:
    """Aggregate daily SKU-level predictions to weekly demand per category.

    The brief's headline metric ("MAPE below 12% on weekly aggregated demand
    predictions across all product categories") is defined at this level, not
    at daily SKU level, so scoring has to aggregate before comparing.

    Args:
        df: A frame with ``date``, ``product_category``, and the actual and
            predicted daily unit columns.
        actual_col: Name of the actual-units column.
        pred_col: Name of the predicted-units column.

    Returns:
        A DataFrame with one row per (product_category, week) holding the
        summed actual and predicted units.
    """
    out = df.copy()
    out["week"] = out["date"].dt.to_period("W").dt.start_time
    return (
        out.groupby(["product_category", "week"])[[actual_col, pred_col]]
        .sum()
        .reset_index()
    )


def run_pipeline(raw_csv_path: str, val_size: float = 0.1,
                 test_size: float = 0.2) -> dict:
    """Run the full cleaning + feature-engineering pipeline end to end.

    Order of operations: load -> daily reindex -> impute -> IQR clip ->
    calendar features -> lag/rolling features -> one-hot encode -> drop
    warm-up rows -> chronological 3-way split -> fit scaler on train ->
    transform all three splits.

    Args:
        raw_csv_path: Path to the input CSV.
        val_size: Fraction of each SKU's rows used for validation.
        test_size: Fraction of each SKU's most recent rows used for test.

    Returns:
        A dict with keys ``train_df``, ``val_df``, ``test_df`` (scaled),
        ``raw_train_df``/``raw_val_df``/``raw_test_df`` (unscaled),
        ``scaler`` and ``feature_columns``.
    """
    df = load_raw(raw_csv_path)
    df = reindex_daily(df)
    df = handle_missing_values(df)
    df = remove_price_outliers(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)

    # Drop the warm-up rows that cannot have a full 30-day window yet.
    df = df.dropna(subset=["roll_mean_30", "roll_std_30", "lag_14",
                           "prev_demand"]).reset_index(drop=True)

    feature_columns = NUMERIC_FEATURES + category_columns(df)

    train_df, val_df, test_df = chronological_split(df, val_size, test_size)
    scaler = fit_scaler(train_df)

    return {
        "train_df": apply_scaler(train_df, scaler),
        "val_df": apply_scaler(val_df, scaler),
        "test_df": apply_scaler(test_df, scaler),
        "raw_train_df": train_df,
        "raw_val_df": val_df,
        "raw_test_df": test_df,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    result = run_pipeline(str(PROJECT_ROOT / "data" / "synthetic_inventory_demand.csv"))
    print("Train:", result["train_df"].shape)
    print("Val:  ", result["val_df"].shape)
    print("Test: ", result["test_df"].shape)
    print("Features:", len(result["feature_columns"]))
