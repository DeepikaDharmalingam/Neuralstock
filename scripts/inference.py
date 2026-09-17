"""
inference.py

Reusable inference pipeline: loads the trained LSTM model and fitted
scaler, then produces a demand forecast for a given product_id from new
input data. Designed to be imported directly by app.py (the Streamlit
dashboard) as well as run standalone from the CLI.

Run standalone:
    python inference.py --product-id P001 --data data/ecommerce_inventory_demand.csv
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import DemandLSTM, set_seeds
from preprocess import (
    NUMERIC_FEATURES,
    add_calendar_features,
    add_lag_and_rolling_features,
    apply_scaler,
    encode_categoricals,
    handle_missing_values,
    load_raw,
    remove_price_outliers,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_LENGTH = 14


def load_artifacts(models_dir: str = None) -> tuple:
    """Load the trained LSTM model weights and fitted scaler from disk.

    Args:
        models_dir: Directory containing lstm_model.pt and scaler.pkl.
            Defaults to <project_root>/models.

    Returns:
        A tuple of (model, scaler). The model is in eval() mode.
    """
    models_dir = models_dir or str(PROJECT_ROOT / "models")
    with open(Path(models_dir) / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = DemandLSTM(input_size=len(NUMERIC_FEATURES))
    state_dict = torch.load(Path(models_dir) / "lstm_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def prepare_features(raw_csv_path: str) -> pd.DataFrame:
    """Run the full feature-engineering pipeline (no scaling) on raw data.

    Args:
        raw_csv_path: Path to the raw CSV (same schema as training data).

    Returns:
        Feature-engineered DataFrame, one-hot encoded, sorted by
        product_id and date, with rows lacking full history dropped.
    """
    df = load_raw(raw_csv_path)
    df = handle_missing_values(df)
    df = remove_price_outliers(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)
    df = df.dropna(subset=["roll_mean_30", "roll_std_30", "lag_14"]).reset_index(drop=True)
    return df


def forecast_for_product(
    product_id: str,
    df: pd.DataFrame,
    model: DemandLSTM,
    scaler,
    horizon: int = 4,
) -> pd.DataFrame:
    """Produce a multi-step-ahead forecast for one product using the trained LSTM.

    Uses a simple recursive strategy: predict one step ahead, append the
    prediction to the feature history (re-deriving lag/rolling features),
    and repeat until `horizon` future points are produced.

    Args:
        product_id: The SKU to forecast for.
        df: Feature-engineered (unscaled) DataFrame covering all products.
        model: A trained, loaded DemandLSTM in eval() mode.
        scaler: The fitted MinMaxScaler used during training.
        horizon: Number of future periods to forecast.

    Returns:
        DataFrame with columns [step, predicted_units_sold] for the
        requested horizon.
    """
    product_df = df[df["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    if len(product_df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Product {product_id} has only {len(product_df)} rows with full "
            f"feature history; need at least {SEQUENCE_LENGTH}."
        )

    history = product_df.copy()
    predictions = []

    for step in range(1, horizon + 1):
        window_df = history.tail(SEQUENCE_LENGTH)
        scaled_window = apply_scaler(window_df, scaler)[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        x = torch.from_numpy(scaled_window).unsqueeze(0)  # [1, seq_len, n_features]

        # The trained LSTM outputs a RESIDUAL (units_sold - roll_mean_7), not
        # an absolute count — see model.py's SlidingWindowDataset docstring
        # for why. Add back the trailing 7-value rolling mean of the most
        # recent known/predicted units_sold to recover a real forecast.
        baseline = float(history["units_sold"].tail(7).mean())
        with torch.no_grad():
            residual_pred = model(x).item()
        pred = max(0.0, baseline + residual_pred)
        predictions.append({"step": step, "predicted_units_sold": round(pred, 1)})

        # Append a synthetic next row so subsequent lag/rolling features
        # reflect this new prediction (recursive forecasting). Each step
        # represents one week (matches app.py's `date + 7*step` display), so
        # advance the date and re-derive every calendar feature from it —
        # leaving them frozen at the last known row would silently feed the
        # model a day_of_week/month/is_weekend that stops changing after
        # step 1, corrupting every later step's dow_sin/cos and month_sin/cos
        # inputs for a multi-step horizon.
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=7)
        next_row = history.iloc[-1].copy()
        next_row["units_sold"] = pred
        next_row["date"] = next_date
        next_row["day_of_week"] = next_date.dayofweek
        next_row["month"] = next_date.month
        history = pd.concat([history, next_row.to_frame().T], ignore_index=True)
        # pd.concat with a mixed-dtype row (built via .copy() on a Series)
        # silently casts numeric columns to `object`, which breaks np.sin
        # inside add_calendar_features below (it needs a numeric dtype to
        # vectorize over). Restore the dtypes that matter before recomputing.
        history["date"] = pd.to_datetime(history["date"])
        history["month"] = history["month"].astype(int)
        history["day_of_week"] = history["day_of_week"].astype(int)
        history = add_calendar_features(
            history.drop(columns=["is_weekend", "month_sin", "month_cos", "dow_sin", "dow_cos"])
        )
        history = add_lag_and_rolling_features(
            history.drop(columns=["lag_7", "lag_14", "roll_mean_7", "roll_std_7",
                                   "roll_mean_30", "roll_std_30", "roll_mean_7_raw"])
        )

    return pd.DataFrame(predictions)


def main() -> None:
    """CLI entry point: load artifacts and print a forecast for one product.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run NeuralStock inference")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "ecommerce_inventory_demand.csv"))
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    set_seeds(42)
    start = time.time()
    model, scaler = load_artifacts(args.models_dir)
    features_df = prepare_features(args.data)
    forecast = forecast_for_product(args.product_id, features_df, model, scaler, args.horizon)
    elapsed = time.time() - start

    print(forecast.to_string(index=False))
    print(f"\nInference completed in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
