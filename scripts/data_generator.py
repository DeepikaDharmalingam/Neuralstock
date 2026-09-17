"""
data_generator.py

Generates a synthetic e-commerce daily sales/inventory dataset that matches
the schema required by the NeuralStock project. Useful for testing the
pipeline end-to-end before the real dataset is available, and for
reproducing results if the original CSV cannot be shared.

Run:
    python data_generator.py --out data/synthetic_inventory_demand.csv
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: The seed value to apply to python's random module and numpy.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)


CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
N_PRODUCTS = 50
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)


def build_product_catalog(n_products: int = N_PRODUCTS) -> pd.DataFrame:
    """Create a static per-product catalog (category, base price, base demand).

    Args:
        n_products: Number of unique SKUs to generate.

    Returns:
        A DataFrame with one row per product_id containing static attributes.
    """
    rows = []
    for i in range(1, n_products + 1):
        category = random.choice(CATEGORIES)
        base_price = {
            "Electronics": np.random.uniform(3000, 60000),
            "Apparel": np.random.uniform(500, 4000),
            "Home": np.random.uniform(800, 15000),
            "Beauty": np.random.uniform(200, 3000),
            "Sports": np.random.uniform(500, 10000),
        }[category]
        base_demand = np.random.uniform(10, 60)
        rows.append(
            {
                "product_id": f"P{i:03d}",
                "product_category": category,
                "base_price": round(base_price, 2),
                "base_demand": base_demand,
                "reorder_point": int(np.random.uniform(15, 50)),
                "supplier_lead_days": int(np.random.uniform(2, 14)),
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(n_products: int = N_PRODUCTS) -> pd.DataFrame:
    """Generate the full daily transaction-level synthetic dataset.

    Args:
        n_products: Number of unique SKUs to simulate.

    Returns:
        A DataFrame with one row per (date, product_id) matching the
        NeuralStock schema: date, product_id, product_category, units_sold,
        unit_price, stock_on_hand, reorder_point, is_promotion, discount_pct,
        day_of_week, month, supplier_lead_days.
    """
    catalog = build_product_catalog(n_products)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    for _, product in catalog.iterrows():
        stock = product["base_demand"] * 20
        for date in dates:
            # Keep the dataset a manageable size: sample ~1 in 6 days per SKU,
            # mirroring the sparsity seen in the real NeuralStock dataset.
            if np.random.rand() > 0.17:
                continue

            dow = date.weekday()
            month = date.month
            is_weekend = dow >= 5
            is_promo = np.random.rand() < 0.06
            discount = np.random.choice([0, 10, 20, 30, 40, 50, 60],
                                         p=[0.82, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]) \
                if is_promo else 0.0

            seasonal = 1 + 0.25 * np.sin(2 * np.pi * month / 12)
            weekend_boost = 1.15 if is_weekend else 1.0
            promo_boost = 1 + (discount / 100.0) * 1.5

            demand = (
                product["base_demand"]
                * seasonal
                * weekend_boost
                * promo_boost
                * np.random.normal(1.0, 0.2)
            )
            demand = max(1, int(round(demand)))

            price = product["base_price"] * (1 - discount / 100.0)
            stock = max(0, stock - demand + np.random.uniform(0, product["base_demand"] * 0.5))

            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "product_id": product["product_id"],
                    "product_category": product["product_category"],
                    "units_sold": demand,
                    "unit_price": round(price, 2),
                    "stock_on_hand": int(stock),
                    "reorder_point": int(product["reorder_point"]),
                    "is_promotion": int(is_promo),
                    "discount_pct": float(discount),
                    "day_of_week": int(dow),
                    "month": int(month),
                    "supplier_lead_days": int(product["supplier_lead_days"]),
                }
            )

    df = pd.DataFrame(records)

    # Inject a small proportion of missing target values, mirroring the
    # real dataset, so the preprocessing pipeline's imputation step is
    # exercised whenever this synthetic data is used.
    missing_mask = np.random.rand(len(df)) < 0.04
    df.loc[missing_mask, "units_sold"] = np.nan

    return df.sort_values(["product_id", "date"]).reset_index(drop=True)


def main() -> None:
    """CLI entry point: generate the dataset and write it to disk.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic NeuralStock dataset")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "synthetic_inventory_demand.csv"),
                         help="Output CSV path")
    parser.add_argument("--n-products", type=int, default=N_PRODUCTS,
                         help="Number of unique SKUs to simulate")
    args = parser.parse_args()

    set_seeds(RANDOM_SEED)
    df = generate_dataset(args.n_products)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows for {df['product_id'].nunique()} products to {args.out}")


if __name__ == "__main__":
    main()
