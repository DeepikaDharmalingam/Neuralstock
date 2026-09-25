"""app.py

NeuralStock Streamlit dashboard. A procurement or warehouse user picks a
product category and SKU, chooses a forecast horizon, and gets a demand
forecast chart, a weekly reorder-alert table and a CSV download.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from inference import (forecast_for_product, load_artifacts, prepare_features,
                       reorder_alerts, weekly_forecast)

st.set_page_config(page_title="NeuralStock Demand Forecast", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "synthetic_inventory_demand.csv")
MODELS_DIR = str(PROJECT_ROOT / "models")
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"


@st.cache_resource
def get_artifacts():
    """Load and cache the trained model, scaler and feature order.

    Returns:
        A ``(model, scaler, feature_columns)`` tuple.
    """
    return load_artifacts(MODELS_DIR)


@st.cache_data
def get_features_df() -> pd.DataFrame:
    """Load and cache the feature-engineered dataset for the session.

    Returns:
        A feature-engineered DataFrame covering all SKUs.
    """
    return prepare_features(DATA_PATH)


@st.cache_data
def get_model_comparison_chart():
    """Build a LSTM vs MLP comparison chart from the saved test metrics.

    Reads ``outputs/metrics.json`` (written by train.py / the notebook) so
    the dashboard always shows the same numbers as the PDF report, with no
    retraining or extra script run needed.

    Layout notes:
        - Extra top margin (``subplots_adjust``/tight_layout rect) so the
          value labels never collide with the subplot titles, which is what
          caused the overlapping "0.980" / "Weekly R2" text before.
        - Value-label y-offset is scaled to each axis's own range instead of
          a fixed 0.1 / 0.01, so it stays readable whatever the metrics are.
        - A shared, slightly larger figsize with more spacing between the
          two subplots (``wspace``) keeps both panels visually balanced.

    Returns:
        A matplotlib Figure, or ``None`` if metrics.json is not present.
    """
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as handle:
        metrics = json.load(handle)

    models = ["LSTM", "MLP"]
    mape_vals = [metrics[m]["weekly"]["mape"] for m in models]
    r2_vals = [metrics[m]["weekly"]["r2"] for m in models]
    colors = ["#4C72B0", "#DD8452"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15)

    ax1.bar(models, mape_vals, color=colors)
    ax1.set_title("Weekly MAPE % (lower is better)", pad=14)
    ax1.set_ylim(0, max(mape_vals) * 1.25)
    for i, v in enumerate(mape_vals):
        ax1.text(i, v + max(mape_vals) * 0.03, f"{v:.2f}",
                 ha="center", va="bottom")

    ax2.bar(models, r2_vals, color=colors)
    ax2.set_title("Weekly R\u00b2 (higher is better)", pad=14)
    ax2.set_ylim(0, 1.08)
    for i, v in enumerate(r2_vals):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

    return fig


def main() -> None:
    """Render the dashboard.

    Returns:
        None.
    """
    st.title("NeuralStock - Deep Learning Demand Forecast")
    st.caption("LSTM-based inventory demand forecasting for e-commerce SKUs")

    try:
        model, scaler, feature_columns = get_artifacts()
        features_df = get_features_df()
    except FileNotFoundError:
        st.error("Model artefacts not found. Run python train.py first to produce "
                 "models/lstm_model.pt, models/scaler.pkl and "
                 "models/feature_columns.json.")
        return

    categories = sorted(features_df["product_category"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Product category", categories)
    with col2:
        skus = sorted(features_df.loc[features_df["product_category"] == category,
                                      "product_id"].unique())
        product_id = st.selectbox("Product (SKU)", skus)
    with col3:
        horizon_weeks = st.slider("Forecast horizon (weeks)", 1, 8, 4)

    min_date = features_df["date"].min().date()
    max_date = features_df["date"].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=180).to_pytimedelta())
    date_range = st.date_input("Historical date range to display",
                               value=(default_start, max_date),
                               min_value=min_date, max_value=max_date)

    if not st.button("Run forecast", type="primary"):
        st.info("Pick a SKU and press **Run forecast**.")
        return

    with st.spinner("Running inference..."):
        forecast = forecast_for_product(product_id, features_df, model, scaler,
                                        feature_columns,
                                        horizon_days=horizon_weeks * 7)

    history = features_df[features_df["product_id"] == product_id].sort_values("date")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        history = history[(history["date"].dt.date >= date_range[0])
                          & (history["date"].dt.date <= date_range[1])]

    st.subheader(f"Daily demand: history and {horizon_weeks}-week forecast - "
                 f"{product_id} ({category})")
    chart_df = pd.concat(
        [
            history[["date", "units_sold"]].rename(columns={"units_sold": "actual"}),
            forecast[["date", "predicted_units_sold"]].rename(
                columns={"predicted_units_sold": "forecast"}),
        ],
        ignore_index=True,
    ).set_index("date")
    st.line_chart(chart_df)

    st.subheader("Weekly forecast")
    st.dataframe(weekly_forecast(forecast), use_container_width=True)

    st.subheader("Reorder alerts")
    full_history = features_df[features_df["product_id"] == product_id].sort_values("date")
    current_stock = float(full_history["stock_on_hand"].iloc[-1])
    reorder_point = float(full_history["reorder_point"].iloc[-1])
    alerts = reorder_alerts(forecast, current_stock, reorder_point)
    st.dataframe(alerts, use_container_width=True)

    if alerts["reorder_needed"].any():
        first_week = alerts.loc[alerts["reorder_needed"], "week"].iloc[0].date()
        st.warning(f"Projected stock crosses the reorder point of {reorder_point:.0f} "
                   f"units in the week of {first_week}. Place a purchase order, "
                   f"allowing for the supplier lead time.")
    else:
        st.success("Projected stock stays above the reorder point across this horizon.")

    st.download_button(
        "Download forecast as CSV",
        data=alerts.to_csv(index=False).encode("utf-8"),
        file_name=f"neuralstock_forecast_{product_id}.csv",
        mime="text/csv",
    )

    comparison_fig = get_model_comparison_chart()
    if comparison_fig is not None:
        st.pyplot(comparison_fig)
    else:
        st.info("Run python train.py to generate outputs/metrics.json before "
                 "this comparison chart can be shown.")


if __name__ == "__main__":
    main()
