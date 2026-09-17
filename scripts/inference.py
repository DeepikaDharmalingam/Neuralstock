"""inference.py

Reusable inference pipeline: loads the trained LSTM and the fitted scaler,
then produces a recursive multi-step demand forecast for a SKU. Imported
directly by ``app.py`` and runnable from the CLI.

The model is trained on a **daily** step, so the forecast recurses one day at
a time and the weekly figures the dashboard shows are sums of those daily
predictions. The previous version advanced seven days per step, which fed the
network a lag-1 feature that was really a lag-7 - a silent mismatch between
training and serving.

Run standalone:
    python inference.py --product-id P001 --horizon-days 28
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from model import DemandLSTM, set_seeds
from preprocess import (add_calendar_features, add_lag_and_rolling_features,
                        apply_scaler, encode_categoricals, handle_missing_values,
                        load_raw, reindex_daily, remove_price_outliers)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_LENGTH = 14

#: Feature columns re-derived after each recursive step.
_CALENDAR_COLUMNS = ["is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos",
                     "quarter_sin", "quarter_cos", "doy_sin", "doy_cos"]
_LAG_COLUMNS = ["prev_demand", "lag_7", "lag_14", "roll_mean_7", "roll_std_7",
                "roll_mean_30", "roll_std_30", "roll_mean_7_raw"]


def load_artifacts(models_dir: str = None) -> Tuple[DemandLSTM, object, List[str]]:
    """Load the trained LSTM, the fitted scaler and the feature order.

    The feature list is read from ``feature_columns.json`` rather than
    recomputed, so the serving-time column order can never drift from what
    the network was trained on.

    Args:
        models_dir: Directory holding lstm_model.pt, scaler.pkl and
            feature_columns.json. Defaults to ``<project_root>/models``.

    Returns:
        A ``(model, scaler, feature_columns)`` tuple, model in eval mode.
    """
    models_dir = Path(models_dir or PROJECT_ROOT / "models")
    with open(models_dir / "scaler.pkl", "rb") as handle:
        scaler = pickle.load(handle)
    with open(models_dir / "feature_columns.json") as handle:
        feature_columns = json.load(handle)

    model = DemandLSTM(input_size=len(feature_columns))
    model.load_state_dict(torch.load(models_dir / "lstm_model.pt", map_location="cpu"))
    model.eval()
    return model, scaler, feature_columns


def prepare_features(raw_csv_path: str) -> pd.DataFrame:
    """Run the full feature-engineering pipeline (unscaled) on raw data.

    Mirrors :func:`preprocess.run_pipeline` exactly up to the scaling step, so
    serving-time features are computed identically to training-time ones.

    Args:
        raw_csv_path: Path to a CSV with the training schema.

    Returns:
        A feature-engineered, one-hot encoded DataFrame sorted by product_id
        and date, with warm-up rows dropped.
    """
    df = load_raw(raw_csv_path)
    df = reindex_daily(df)
    df = handle_missing_values(df)
    df = remove_price_outliers(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)
    return df.dropna(subset=["roll_mean_30", "roll_std_30", "lag_14",
                             "prev_demand"]).reset_index(drop=True)


def forecast_for_product(product_id: str, df: pd.DataFrame, model: DemandLSTM,
                         scaler, feature_columns: List[str],
                         horizon_days: int = 28) -> pd.DataFrame:
    """Produce a recursive daily forecast for one SKU.

    Strategy: predict day t+1, append it to the history, re-derive every lag,
    rolling and calendar feature from the extended history, and repeat. Error
    compounds across steps, which is why the dashboard reports weekly sums
    rather than individual far-out days.

    Args:
        product_id: SKU to forecast.
        df: Unscaled feature-engineered frame covering all SKUs.
        model: A trained DemandLSTM in eval mode.
        scaler: The MinMaxScaler fitted during training.
        feature_columns: Feature order from ``feature_columns.json``.
        horizon_days: Number of future days to forecast (28 = 4 weeks).

    Returns:
        A DataFrame with columns ``date``, ``predicted_units_sold`` and
        ``week``, one row per forecast day.

    Raises:
        ValueError: If the SKU has fewer than ``SEQUENCE_LENGTH`` usable rows.
    """
    history = df[df["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    if len(history) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Product {product_id} has only {len(history)} rows with full feature "
            f"history; at least {SEQUENCE_LENGTH} are required."
        )

    predictions = []
    for _ in range(horizon_days):
        window = apply_scaler(history.tail(SEQUENCE_LENGTH), scaler)
        x = torch.from_numpy(window[feature_columns].to_numpy(dtype=np.float32)).unsqueeze(0)

        # The network emits a RESIDUAL against the trailing 7-day mean (see
        # SlidingWindowDataset); add that baseline back for real units.
        baseline = float(history["units_sold"].tail(7).mean())
        with torch.no_grad():
            residual = model(x).item()
        prediction = max(0.0, baseline + residual)

        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        predictions.append({"date": next_date,
                            "predicted_units_sold": round(prediction, 2)})

        next_row = history.iloc[-1].copy()
        next_row["date"] = next_date
        next_row["units_sold"] = prediction
        next_row["day_of_week"] = next_date.dayofweek
        next_row["month"] = next_date.month
        history = pd.concat([history, next_row.to_frame().T], ignore_index=True)

        # Concatenating a Series row casts numeric columns to object, which
        # breaks the vectorised np.sin calls below. Restore dtypes first.
        history["date"] = pd.to_datetime(history["date"])
        history["month"] = history["month"].astype(int)
        history["day_of_week"] = history["day_of_week"].astype(int)
        history["units_sold"] = history["units_sold"].astype(float)

        history = add_calendar_features(history.drop(columns=_CALENDAR_COLUMNS))
        history = add_lag_and_rolling_features(history.drop(columns=_LAG_COLUMNS))

    forecast = pd.DataFrame(predictions)
    forecast["week"] = forecast["date"].dt.to_period("W").dt.start_time
    return forecast


def weekly_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily forecast into weekly totals.

    Args:
        forecast: Output of :func:`forecast_for_product`.

    Returns:
        A DataFrame with ``week`` and ``forecast_units`` columns.
    """
    return (forecast.groupby("week")["predicted_units_sold"]
            .sum().round(1).reset_index(name="forecast_units"))


def reorder_alerts(forecast: pd.DataFrame, stock_on_hand: float,
                   reorder_point: float) -> pd.DataFrame:
    """Flag the weeks where projected stock falls through the reorder point.

    Args:
        forecast: Output of :func:`forecast_for_product`.
        stock_on_hand: Current opening inventory for the SKU.
        reorder_point: Replenishment threshold for the SKU.

    Returns:
        A DataFrame with week, forecast units, projected closing stock and a
        boolean ``reorder_needed`` flag.
    """
    weekly = weekly_forecast(forecast)
    projected, rows = float(stock_on_hand), []
    for _, row in weekly.iterrows():
        projected -= row["forecast_units"]
        rows.append({
            "week": row["week"],
            "forecast_units": row["forecast_units"],
            "projected_stock": round(projected, 1),
            "reorder_needed": bool(projected <= reorder_point),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point: load artefacts and print a forecast for one SKU.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run NeuralStock inference")
    parser.add_argument("--product-id", required=True)
    default_data = PROJECT_ROOT / "data" / "synthetic_inventory_demand.csv"
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--horizon-days", type=int, default=28)
    args = parser.parse_args()

    set_seeds(42)
    start = time.time()
    model, scaler, feature_columns = load_artifacts(args.models_dir)
    features = prepare_features(args.data)
    forecast = forecast_for_product(args.product_id, features, model, scaler,
                                    feature_columns, args.horizon_days)
    elapsed = time.time() - start

    print(weekly_forecast(forecast).to_string(index=False))
    print(f"\nInference completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
