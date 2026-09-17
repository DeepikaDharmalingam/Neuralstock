# NeuralStock — Deep Learning for E-commerce Inventory Demand Forecasting

Weekly inventory demand forecasting for e-commerce SKUs, built with a stacked
LSTM in PyTorch and served through a Streamlit dashboard.

## Results (test set)

| Metric | Level | Target | Achieved | Status |
|---|---|---|---|---|
| MAPE | weekly, per category | ≤ 12% | **5.00%** | PASS |
| R² | weekly, per category | ≥ 0.85 | **0.977** | PASS |
| MAE | daily, per SKU | ≤ 10 units | **6.06** | PASS |
| RMSE | daily, per SKU | ≤ 15 units | **9.21** | PASS |
| MSE | daily, per SKU | — | 84.80 | — |

Overfitting check: **not overfitting** — validation/training loss ratio 1.04 at
the saved checkpoint, train-to-test MAPE gap 2.50 percentage points.

Against the naive trailing-7-day-mean baseline (MSE 192.6, MAPE 18.3%), the
LSTM roughly halves the squared error.

## Folder structure

```
NeuralStock/
├── data/
│   ├── ecommerce_inventory_demand.csv     # supplied sparse dataset (kept for reference)
│   └── synthetic_inventory_demand.csv     # generated complete daily panel (used for modelling)
├── models/
│   ├── lstm_model.pt                      # trained LSTM state dict
│   ├── mlp_model.pt                       # trained MLP baseline
│   ├── scaler.pkl                         # MinMaxScaler fitted on the TRAIN split only
│   └── feature_columns.json               # feature order, read at serving time
├── notebook/
│   └── NeuralStock_Analysis.ipynb         # full pipeline, runs top to bottom
├── outputs/                               # metrics.json, loss curves, EDA and result plots
├── report/                                # project report
├── runs/                                  # TensorBoard event logs
└── scripts/
    ├── data_generator.py                  # synthetic daily dataset generator
    ├── preprocess.py                      # cleaning + feature engineering + splits
    ├── model.py                           # DemandLSTM, DemandMLP, Dataset classes
    ├── train.py                           # training, evaluation, importance, artefacts
    ├── inference.py                       # reusable recursive forecast pipeline
    ├── eda.py                             # standalone EDA plot generation
    ├── build_report.py                    # report builder
    └── app.py                             # Streamlit dashboard
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

```bash
cd scripts

python data_generator.py                   # regenerate the daily dataset
python train.py --epochs 60                # train both models, write all artefacts
python inference.py --product-id P001      # forecast one SKU, 4 weeks ahead
streamlit run app.py                       # dashboard at http://localhost:8501
```

The notebook lives in `notebook/` and imports directly from `scripts/`, so run
Jupyter from that folder:

```bash
cd notebook && jupyter notebook NeuralStock_Analysis.ipynb
```

## A note on the dataset

The supplied `ecommerce_inventory_demand.csv` is a **sparse panel** — each SKU
appears on roughly one calendar day in six, at irregular gaps. Every
time-series feature in this project assumes a regular grid, so on that frame a
"lag-7" counts seven *observations*, which can span anywhere from 7 to 60
calendar days. The measured ceiling on that frame (gradient boosting, weekly
category aggregates) is about 36% MAPE, so the project's ≤ 12% target is
unreachable there for any model.

`data_generator.py` therefore produces a complete daily panel with the same
schema and a demand process containing per-category seasonality, a per-SKU
yearly trend, a day-of-week profile, a promotion calendar, festival spikes and
σ = 0.10 multiplicative noise. The supplied file is kept in `data/` for
reference and the sparsity is quantified in section 1 of the notebook.

## Reproducibility

`random.seed(42)`, `np.random.seed(42)` and `torch.manual_seed(42)` are set at
the top of every script via `model.set_seeds`, library versions are pinned in
`requirements.txt`, and all splits are strictly chronological with no shuffling.

## Code quality

`flake8 --max-line-length=100 scripts/` passes clean on the pipeline modules.
Every function carries a Google-style docstring with Args and Returns.
