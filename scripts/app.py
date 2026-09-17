"""
app.py

NeuralStock Streamlit dashboard: lets a warehouse/procurement user pick a
product category and date range, runs the trained LSTM model, and shows a
demand forecast chart with reorder alerts and a CSV download button.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import forecast_for_product, load_artifacts, prepare_features

st.set_page_config(page_title="NeuralStock Demand Forecast", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "ecommerce_inventory_demand.csv")
MODELS_DIR = str(PROJECT_ROOT / "models")


@st.cache_resource
def get_model_and_scaler():
    """Load and cache the trained model and scaler for the session.

    Returns:
        (model, scaler) tuple.
    """
    return load_artifacts(MODELS_DIR)


@st.cache_data
def get_features_df() -> pd.DataFrame:
    """Load and cache the feature-engineered dataset for the session.

    Returns:
        Feature-engineered DataFrame covering all products.
    """
    return prepare_features(DATA_PATH)


def main() -> None:
    """Render the Streamlit dashboard.

    Returns:
        None.
    """
    st.title("📦 NeuralStock — Deep Learning Demand Forecast")
    st.caption("LSTM-based weekly inventory demand forecasting for e-commerce SKUs")

    try:
        model, scaler = get_model_and_scaler()
        features_df = get_features_df()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python train.py` first to "
            "produce models/lstm_model.pt and models/scaler.pkl."
        )
        return

    categories = sorted(
        [c.replace("category_", "") for c in features_df.columns if c.startswith("category_")]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Product category", categories)
    with col2:
        products_in_category = sorted(
            features_df.loc[features_df[f"category_{category}"] == 1, "product_id"].unique()
        )
        product_id = st.selectbox("Product (SKU)", products_in_category)
    with col3:
        horizon_weeks = st.slider("Forecast horizon (steps)", min_value=1, max_value=8, value=4)

    date_range = st.date_input(
        "Historical date range to display",
        value=(
            features_df["date"].min().date(),
            features_df["date"].max().date(),
        ),
    )

    if st.button("Run forecast", type="primary"):
        with st.spinner("Running inference..."):
            forecast_df = forecast_for_product(
                product_id, features_df, model, scaler, horizon=horizon_weeks
            )

        history = features_df[features_df["product_id"] == product_id].sort_values("date")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            history = history[
                (history["date"].dt.date >= date_range[0])
                & (history["date"].dt.date <= date_range[1])
            ]

        last_date = history["date"].max() if len(history) else pd.Timestamp.today()
        forecast_df["date"] = [last_date + timedelta(days=7 * i) for i in forecast_df["step"]]

        st.subheader(f"Forecast for {product_id} ({category})")

        chart_df = pd.concat(
            [
                history[["date", "units_sold"]].rename(columns={"units_sold": "actual"}),
                forecast_df[["date", "predicted_units_sold"]].rename(
                    columns={"predicted_units_sold": "forecast"}
                ),
            ],
            ignore_index=True,
        ).set_index("date")
        st.line_chart(chart_df)

        st.subheader("Reorder alerts")
        current_stock = history["stock_on_hand"].iloc[-1] if len(history) else None
        reorder_point = history["reorder_point"].iloc[-1] if len(history) else None
        alert_rows = []
        projected_stock = current_stock
        for _, row in forecast_df.iterrows():
            projected_stock = max(0, projected_stock - row["predicted_units_sold"]) if projected_stock is not None else None
            alert_rows.append(
                {
                    "step": row["step"],
                    "date": row["date"].date(),
                    "predicted_units_sold": row["predicted_units_sold"],
                    "projected_stock": projected_stock,
                    "below_reorder_point": (
                        projected_stock is not None and reorder_point is not None
                        and projected_stock < reorder_point
                    ),
                }
            )
        alerts_df = pd.DataFrame(alert_rows)
        st.dataframe(alerts_df, use_container_width=True)

        if alerts_df["below_reorder_point"].any():
            st.warning(
                f"⚠️ Projected stock falls below the reorder point of {reorder_point} "
                f"units within the forecast horizon. Consider placing a purchase order."
            )
        else:
            st.success("✅ Projected stock stays above the reorder point for this horizon.")

        csv_bytes = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast as CSV",
            data=csv_bytes,
            file_name=f"neuralstock_forecast_{product_id}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
