"""
train.py

Trains the DemandLSTM and DemandMLP models on the preprocessed NeuralStock
dataset, logs metrics to TensorBoard, evaluates both against a naive
baseline, and saves the winning model's weights plus the fitted scaler.

Run:
    python train.py --data data/ecommerce_inventory_demand.csv --epochs 40

Outputs:
    models/lstm_model.pt   (PyTorch state dict)
    models/mlp_model.pt    (PyTorch state dict, baseline)
    models/scaler.pkl      (fitted MinMaxScaler, joblib)
    runs/                  (TensorBoard event logs)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import DemandLSTM, DemandMLP, SlidingWindowDataset, TabularDataset, set_seeds
from preprocess import bridge_train_test_for_windows, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_LENGTH = 14


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE, guarding against division by zero.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (e.g. 11.4 means 11.4%).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    writer: SummaryWriter,
    tag: str,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> nn.Module:
    """Train a single model with Adam + ReduceLROnPlateau, logging to TensorBoard.

    Args:
        model: The PyTorch model to train (modified in place).
        train_loader: DataLoader yielding (X, y, baseline) triples — see
            SlidingWindowDataset / TabularDataset. `baseline` is unused
            here; loss is always computed against whatever `y` the
            dataset provides (residual for the LSTM, raw for the MLP).
        val_loader: DataLoader over the validation/test set, same shape.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        writer: TensorBoard SummaryWriter.
        tag: Prefix used for TensorBoard scalar names (e.g. "LSTM" or "MLP").
        device: torch device to train on.
        grad_clip_norm: Max gradient norm (clip_grad_norm_). LSTMs are
            prone to occasional large gradients through the recurrence;
            clipping keeps a single bad batch from derailing training.

    Returns:
        The trained model (best validation-loss checkpoint restored).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y, _baseline in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, _baseline in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_loss += criterion(preds, y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        writer.add_scalar(f"{tag}/train_loss", train_loss, epoch)
        writer.add_scalar(f"{tag}/val_loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             predict_residual: bool = False) -> dict:
    """Evaluate a trained model on a held-out DataLoader.

    Args:
        model: A trained PyTorch model.
        loader: DataLoader yielding (X, y, baseline) triples.
        device: torch device.
        predict_residual: Must match how the loader's dataset was built.
            When True, both the model's output and `y` are residuals
            (target - baseline); this adds `baseline` back onto each
            before scoring, so metrics are always reported in real
            units_sold units regardless of what the model was trained on.

    Returns:
        Dict with keys mae, rmse, mape, r2, y_true, y_pred (the last two
        as plain lists, so the dict is JSON-serialisable for plotting
        or reporting later without needing to re-run inference).
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, baseline in loader:
            X = X.to(device)
            preds = model(X).cpu().numpy()
            targets = y.numpy()
            baseline = baseline.numpy()
            if predict_residual:
                preds = preds + baseline
                targets = targets + baseline
            all_preds.append(preds)
            all_targets.append(targets)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def naive_baseline_metrics(loader: DataLoader, predict_residual: bool = False) -> dict:
    """Compute metrics for the naive 'predict = roll_mean_7' baseline.

    Forecasting the trailing 7-day rolling mean of units_sold is the naive
    comparator every trained model should beat. Both Dataset classes hand
    this value back as `baseline` on every batch, so this works
    identically for the LSTM's sliding-window loader and the MLP's
    tabular loader.

    Args:
        loader: A DataLoader yielding (X, y, baseline) triples.
        predict_residual: Whether `y` on this loader is a residual
            (target - baseline) rather than the raw target; set this to
            match the loader (True for the LSTM's loader, False for the
            MLP's).

    Returns:
        Dict with keys mae, rmse, mape, r2 for the naive baseline.
    """
    all_preds, all_targets = [], []
    for _X, y, baseline in loader:
        baseline = baseline.numpy()
        target = y.numpy() + baseline if predict_residual else y.numpy()
        all_preds.append(baseline)
        all_targets.append(target)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics_json(metrics_by_model: dict, out_path: str) -> None:
    """Persist final test-set metrics to a JSON file for report generation.

    Args:
        metrics_by_model: Dict mapping model name -> metrics dict (as
            returned by `evaluate` / `naive_baseline_metrics`).
        out_path: File path to write the JSON to.

    Returns:
        None.
    """
    # Strip the raw prediction arrays before saving the summary file —
    # those are only needed transiently to build the actual-vs-predicted
    # plot, not for the report's metrics table.
    summary = {
        name: {k: v for k, v in m.items() if k not in ("y_true", "y_pred")}
        for name, m in metrics_by_model.items()
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def save_actual_vs_predicted_plot(lstm_metrics: dict, out_path: str, n_points: int = 150) -> None:
    """Plot the trained LSTM's real test-set predictions against actuals.

    Args:
        lstm_metrics: The dict returned by `evaluate` for the LSTM model
            (must contain y_true and y_pred lists).
        out_path: File path (PNG) to save the plot to.
        n_points: Number of test-set points to display.

    Returns:
        None.
    """
    y_true = np.array(lstm_metrics["y_true"])[:n_points]
    y_pred = np.array(lstm_metrics["y_pred"])[:n_points]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(y_true)), y_true, label="Actual", color="#333333")
    ax.plot(range(len(y_pred)), y_pred, label="Predicted (LSTM)", color="#C44E52", alpha=0.85)
    ax.set_title(f"Actual vs Predicted units_sold — LSTM (test set, first {len(y_true)} rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """CLI entry point: run preprocessing, train both models, save artifacts.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Train NeuralStock LSTM & MLP models")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "ecommerce_inventory_demand.csv"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--logs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_dir).mkdir(parents=True, exist_ok=True)

    pipeline_out = run_pipeline(args.data)
    train_df, test_df = pipeline_out["train_df"], pipeline_out["test_df"]
    feature_columns = pipeline_out["feature_columns"]
    scaler = pipeline_out["scaler"]

    with open(Path(args.models_dir) / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ---- LSTM (sequence) ----
    # Bridge each product's trailing TRAINING rows onto its test split so
    # every test row gets a window (see bridge_train_test_for_windows'
    # docstring) — otherwise only the first ~sequence_length rows per
    # product would be usable and the LSTM's test set would silently be a
    # different, smaller sample than the MLP's.
    test_bridged_df = bridge_train_test_for_windows(train_df, test_df, args.sequence_length)

    train_seq_ds = SlidingWindowDataset(
        train_df, feature_columns, sequence_length=args.sequence_length, predict_residual=True,
    )
    test_seq_ds = SlidingWindowDataset(
        test_bridged_df, feature_columns, sequence_length=args.sequence_length, predict_residual=True,
    )
    train_seq_loader = DataLoader(train_seq_ds, batch_size=args.batch_size, shuffle=True)
    test_seq_loader = DataLoader(test_seq_ds, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter(log_dir=str(Path(args.logs_dir) / "neuralstock"))

    lstm_model = DemandLSTM(input_size=len(feature_columns))
    lstm_model = train_one_model(
        lstm_model, train_seq_loader, test_seq_loader,
        epochs=args.epochs, lr=args.lr, writer=writer, tag="LSTM", device=device,
    )
    lstm_metrics = evaluate(lstm_model, test_seq_loader, device, predict_residual=True)
    torch.save(lstm_model.state_dict(), Path(args.models_dir) / "lstm_model.pt")

    # ---- MLP (tabular baseline) ----
    train_tab_ds = TabularDataset(train_df, feature_columns)
    test_tab_ds = TabularDataset(test_df, feature_columns)
    train_tab_loader = DataLoader(train_tab_ds, batch_size=args.batch_size, shuffle=True)
    test_tab_loader = DataLoader(test_tab_ds, batch_size=args.batch_size, shuffle=False)

    mlp_model = DemandMLP(input_size=len(feature_columns))
    mlp_model = train_one_model(
        mlp_model, train_tab_loader, test_tab_loader,
        epochs=args.epochs, lr=args.lr, writer=writer, tag="MLP", device=device,
    )
    mlp_metrics = evaluate(mlp_model, test_tab_loader, device, predict_residual=False)
    torch.save(mlp_model.state_dict(), Path(args.models_dir) / "mlp_model.pt")

    naive_metrics = naive_baseline_metrics(test_tab_loader, predict_residual=False)

    print("\n=== Final Test-Set Metrics ===")
    for name, metrics in [("LSTM", lstm_metrics), ("MLP", mlp_metrics), ("Naive baseline", naive_metrics)]:
        print(f"{name:15s}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.3f}")

    save_metrics_json(
        {"LSTM": lstm_metrics, "MLP": mlp_metrics, "Naive baseline": naive_metrics},
        str(Path(args.outputs_dir) / "metrics.json"),
    )
    save_actual_vs_predicted_plot(
        lstm_metrics, str(Path(args.outputs_dir) / "actual_vs_predicted.png")
    )

    writer.close()
    print(f"\nSaved lstm_model.pt, mlp_model.pt, scaler.pkl to {args.models_dir}/")
    print(f"Saved metrics.json and actual_vs_predicted.png to {args.outputs_dir}/")
    print(f"TensorBoard logs at {args.logs_dir}/  (run: tensorboard --logdir {args.logs_dir})")


if __name__ == "__main__":
    main()
