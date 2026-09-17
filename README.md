# NeuralStock: Deep Learning for E-commerce Inventory Demand Forecasting

Predicts weekly per-SKU demand using a stacked LSTM (with an MLP baseline),
enabling smarter procurement, stockout prevention, and reorder alerts —
delivered as a reusable inference pipeline plus a Streamlit dashboard.

## Folder Structure

```
NeuralStock/
├── data/
│   └── ecommerce_inventory_demand.csv     # raw dataset
├── models/                                # empty until you run train.py
├── notebook/
│   └── NeuralStock_Analysis.ipynb         # EDA → preprocessing → modelling → insights
├── scripts/
│   ├── data_generator.py                  # synthetic dataset (same schema)
│   ├── preprocess.py                      # cleaning + feature engineering
│   ├── model.py                           # DemandLSTM, DemandMLP, Dataset classes
│   ├── train.py                           # training loop + TensorBoard + metrics.json
│   ├── inference.py                       # reusable forecasting pipeline
│   ├── app.py                             # Streamlit dashboard
│   ├── eda.py                             # generates EDA plots from the real data
│   └── build_report.py                    # assembles the PDF report from YOUR results
├── report/                                # empty until you run build_report.py
├── outputs/                               # empty until you run eda.py / train.py
├── requirements.txt
├── .gitignore
└── README.md
```

No result files, plots, or the report PDF are pre-generated. Everything in
`outputs/`, `models/`, and `report/` is produced by running the scripts
below on your own machine — the numbers and charts you submit are genuinely
yours.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

All scripts resolve `data/`, `models/`, `outputs/`, and `report/` relative
to the project root automatically, so you can run them either from the
project root (`python scripts/train.py`) or from inside `scripts/`
(`cd scripts && python train.py`) — both work identically.

1. **Dataset** is already at `data/ecommerce_inventory_demand.csv`.
   Need a synthetic one instead (e.g. to test the pipeline without the
   original file)?
   ```bash
   python scripts/data_generator.py
   ```

2. **Generate EDA plots** from the real data (no PyTorch required for this step):
   ```bash
   python scripts/eda.py
   ```
   Writes 5 plots to `outputs/`.

3. **Train both models** (produces `models/lstm_model.pt`, `models/mlp_model.pt`,
   `models/scaler.pkl`, `outputs/metrics.json`, and `outputs/actual_vs_predicted.png`):
   ```bash
   python scripts/train.py --epochs 40
   ```
   View training curves with:
   ```bash
   tensorboard --logdir runs
   ```
   Take a screenshot of the loss curves for the report (required by the
   project guidelines — `build_report.py` leaves a placeholder for it).

4. **Build the PDF report** from your real metrics and plots:
   ```bash
   python scripts/build_report.py
   ```
   This will refuse to run and tell you exactly what's missing if you
   haven't completed steps 2–3 yet — it never fabricates numbers.

5. **Run inference from the CLI**:
   ```bash
   python scripts/inference.py --product-id P001 --horizon 4
   ```

6. **Launch the Streamlit dashboard**:
   ```bash
   cd scripts && streamlit run app.py
   ```
   Select a category and SKU, choose a forecast horizon, and download the
   forecast + reorder-alert table as CSV.

7. **Explore the notebook**:
   ```bash
   jupyter notebook notebook/NeuralStock_Analysis.ipynb
   ```
   Run all cells top-to-bottom — they call the same functions as the
   scripts above, against the same real data.

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| date | date | Transaction date (daily) |
| product_id | string | Unique SKU identifier (P001–P050) |
| product_category | string | Electronics, Apparel, Home, Beauty, Sports |
| units_sold | int | Daily units sold (**target**) |
| unit_price | float | Selling price per unit (INR) |
| stock_on_hand | int | Opening inventory level for the day |
| reorder_point | int | Threshold at which replenishment is triggered |
| is_promotion | bool | 1 if a promotional event was active |
| discount_pct | float | Discount percentage applied (0–60) |
| day_of_week | int | 0=Monday … 6=Sunday |
| month | int | Calendar month (1–12) |
| supplier_lead_days | int | Average lead time from supplier (days) |

## Reproducibility

All scripts call `set_seeds(42)` (Python `random`, NumPy, and `torch`) at
the top of execution. Library versions are pinned in `requirements.txt`.

## No Data Leakage

`MinMaxScaler` is fit **only** on the training split (`fit_scaler` in
`preprocess.py`) and applied separately to test/inference data. Train/test
splits are strictly chronological per product — no shuffling.

## Ethical Considerations

Over-reliance on automated reorder suggestions can itself cause supply
chain disruptions if the model silently fails (e.g. during a demand shock
the training data never saw). The dashboard's reorder alerts should
support, not replace, a human buyer's judgment. **Cold-start limitation**:
new SKUs with fewer than 14 historical observations cannot be forecast by
the LSTM until enough history accumulates.

## Cloud Deployment

Deploy `scripts/app.py` to your preferred provider:
- **AWS EC2 / GCP Compute Engine / Azure VM**: `pip install -r
  requirements.txt`, then `streamlit run scripts/app.py --server.port 8501
  --server.address 0.0.0.0`, open the port in your security group, and
  share the instance's public IP:8501.
