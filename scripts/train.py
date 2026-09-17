"""train.py

Trains :class:`DemandLSTM` and :class:`DemandMLP` on the preprocessed
NeuralStock data, logs to TensorBoard, scores both against a naive baseline
at daily SKU level *and* at weekly category level, checks for overfitting,
computes permutation feature importance, and saves all artefacts.

Run:
    python train.py --data ../data/synthetic_inventory_demand.csv --epochs 40

Outputs:
    models/lstm_model.pt          PyTorch state dict
    models/mlp_model.pt           PyTorch state dict (baseline)
    models/scaler.pkl             fitted MinMaxScaler
    models/feature_columns.json   feature order, for the inference pipeline
    outputs/metrics.json          daily + weekly metrics for every model
    outputs/feature_importance.csv
    runs/                         TensorBoard event logs
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from model import (DemandLSTM, DemandMLP, SlidingWindowDataset, TabularDataset,
                   set_seeds)
from preprocess import aggregate_weekly, bridge_for_windows, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQUENCE_LENGTH = 14

#: Acceptance thresholds from the project brief, for the pass/fail table.
METRIC_TARGETS = {"mape": 12.0, "rmse": 15.0, "mae": 10.0, "r2": 0.85}


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE as a percentage, skipping zero-valued actuals.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (11.4 means 11.4%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the full metric set for one set of predictions.

    MSE is reported alongside RMSE because MSE is the quantity the training
    loss actually minimises - quoting RMSE alone hides the scale on which the
    optimiser was scored, and the brief's loss function is MSE.

    Args:
        y_true: Ground-truth values, in real units.
        y_pred: Predicted values, in real units.

    Returns:
        Dict with keys mse, rmse, mae, mape, r2.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    writer,
    tag: str,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    patience: int = 10,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train one model with Adam + ReduceLROnPlateau and early stopping.

    Model selection uses the **validation** split only. The original pipeline
    selected its best checkpoint on the test split, which leaks test
    information into training and inflates the reported score.

    Args:
        model: The PyTorch model to train (modified in place).
        train_loader: Loader yielding ``(X, y, baseline, row_index)``.
        val_loader: Loader over the validation split, same tuple shape.
        epochs: Maximum number of epochs.
        lr: Initial learning rate for Adam.
        writer: TensorBoard SummaryWriter (or None to skip logging).
        tag: Prefix for TensorBoard scalar names, e.g. "LSTM".
        device: Torch device to train on.
        grad_clip_norm: Max gradient norm. LSTMs occasionally produce large
            gradients through the recurrence; clipping stops one bad batch
            from derailing training.
        patience: Epochs without validation improvement before stopping.
        weight_decay: L2 penalty on the weights. A small amount here plus
            dropout is what keeps the validation curve from separating from
            the training curve on this dataset.

    Returns:
        A ``(model, history)`` tuple. ``model`` has the best-validation
        weights restored; ``history`` is a DataFrame of per-epoch train and
        validation MSE loss, used for the overfitting analysis.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.MSELoss()

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y, _baseline, _idx in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, _baseline, _idx in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if writer is not None:
            writer.add_scalar(f"{tag}/train_loss", train_loss, epoch)
            writer.add_scalar(f"{tag}/val_loss", val_loss, epoch)
            writer.add_scalar(f"{tag}/lr", optimizer.param_groups[0]["lr"], epoch)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_loss:8.4f}  val_mse={val_loss:8.4f}")

        if bad_epochs >= patience:
            print(f"[{tag}] early stop at epoch {epoch} "
                  f"(no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            predict_residual: bool = True) -> pd.DataFrame:
    """Run inference and return predictions in real units, with row indices.

    Args:
        model: A trained PyTorch model.
        loader: Loader yielding ``(X, y, baseline, row_index)``. Must not be
            shuffled, though the returned row index makes order irrelevant.
        device: Torch device.
        predict_residual: Must match how the loader's dataset was built. When
            True, ``baseline`` is added back onto both prediction and target
            so everything is scored in real units_sold.

    Returns:
        A DataFrame indexed by the source row index with columns
        ``y_true``, ``y_pred`` and ``baseline``.
    """
    model.eval()
    preds, trues, bases, idxs = [], [], [], []
    with torch.no_grad():
        for X, y, baseline, row_index in loader:
            out = model(X.to(device)).cpu().numpy()
            target = y.numpy()
            base = baseline.numpy()
            if predict_residual:
                out = out + base
                target = target + base
            preds.append(out)
            trues.append(target)
            bases.append(base)
            idxs.append(row_index.numpy())

    return pd.DataFrame(
        {
            "y_true": np.concatenate(trues),
            "y_pred": np.clip(np.concatenate(preds), 0, None),
            "baseline": np.concatenate(bases),
        },
        index=np.concatenate(idxs),
    )


def score_daily_and_weekly(pred_df: pd.DataFrame, source_df: pd.DataFrame,
                           pred_col: str = "y_pred") -> Dict[str, dict]:
    """Score predictions at daily SKU level and weekly category level.

    The brief defines its headline MAPE on "weekly aggregated demand
    predictions across all product categories", while its MAE and RMSE
    thresholds (<= 10, <= 15 units) are only meaningful at daily SKU scale.
    Both levels are therefore reported.

    Args:
        pred_df: Output of :func:`predict`, indexed by source row index.
        source_df: The unscaled split frame the predictions came from; must
            carry ``date`` and ``product_category``.
        pred_col: Which prediction column of ``pred_df`` to score.

    Returns:
        Dict with keys ``daily`` and ``weekly``, each a metric dict.
    """
    joined = pred_df.join(source_df[["date", "product_category"]], how="inner")
    daily = regression_metrics(joined["y_true"], joined[pred_col])
    weekly_df = aggregate_weekly(joined, "y_true", pred_col)
    weekly = regression_metrics(weekly_df["y_true"], weekly_df[pred_col])
    return {"daily": daily, "weekly": weekly}


def overfitting_report(history: pd.DataFrame, train_metrics: dict,
                       test_metrics: dict) -> dict:
    """Summarise the evidence for or against overfitting.

    The brief requires proof that the model is not overfitting. Two checks
    are used, both taken at the best-validation epoch (the checkpoint that is
    actually saved): the validation-to-training loss ratio, and the gap in
    MAPE between the training and test splits. MAPE is used rather than MAE
    because demand trends upwards over the two-year window, so absolute
    errors on the later test period are larger for reasons that have nothing
    to do with overfitting.

    Args:
        history: Per-epoch loss history from :func:`train_one_model`.
        train_metrics: Daily metric dict computed on the training split.
        test_metrics: Daily metric dict computed on the test split.

    Returns:
        Dict with the loss ratio, the MAPE gap in percentage points, and a
        verdict string. A ratio near 1.0 and a MAPE gap under ~4 points
        indicate the model generalises rather than memorises.
    """
    best = history.loc[history["val_loss"].idxmin()]
    loss_ratio = float(best["val_loss"] / max(best["train_loss"], 1e-9))
    mape_gap = float(test_metrics["mape"] - train_metrics["mape"])
    overfit = loss_ratio > 1.35 or mape_gap > 4.0
    return {
        "best_epoch": int(best["epoch"]),
        "train_loss_at_best": float(best["train_loss"]),
        "val_loss_at_best": float(best["val_loss"]),
        "val_over_train_loss_ratio": loss_ratio,
        "train_mape": train_metrics["mape"],
        "test_mape": test_metrics["mape"],
        "test_minus_train_mape_points": mape_gap,
        "verdict": "overfitting" if overfit else "not overfitting",
    }


def permutation_importance(model: nn.Module, dataset, feature_columns: List[str],
                           device: torch.device, batch_size: int = 512,
                           n_repeats: int = 2, seed: int = 42) -> pd.DataFrame:
    """Rank features by the damage done when each one is shuffled.

    Permutation importance is model-agnostic and works for both the sequence
    and tabular models: shuffle one feature column across samples (and across
    all time steps, for the LSTM), re-score, and measure how much MAE rises.

    Args:
        model: A trained model.
        dataset: A SlidingWindowDataset or TabularDataset instance.
        feature_columns: Feature names in the same order as the dataset's
            feature axis.
        device: Torch device.
        batch_size: Inference batch size.
        n_repeats: Number of shuffles averaged per feature.
        seed: Seed for the shuffle.

    Returns:
        A DataFrame with columns ``feature`` and ``mae_increase``, sorted
        descending by importance.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    X = dataset.X.copy()
    y_real = dataset.y + dataset.baseline if dataset.predict_residual else dataset.y

    def _score(matrix: np.ndarray) -> float:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                batch = torch.from_numpy(matrix[start:start + batch_size]).to(device)
                outputs.append(model(batch).cpu().numpy())
        preds = np.concatenate(outputs)
        if dataset.predict_residual:
            preds = preds + dataset.baseline
        return float(mean_absolute_error(y_real, preds))

    baseline_mae = _score(X)
    rows = []
    for f_idx, name in enumerate(feature_columns):
        deltas = []
        for _ in range(n_repeats):
            permuted = X.copy()
            order = rng.permutation(len(permuted))
            if permuted.ndim == 3:
                permuted[:, :, f_idx] = permuted[order][:, :, f_idx]
            else:
                permuted[:, f_idx] = permuted[order][:, f_idx]
            deltas.append(_score(permuted) - baseline_mae)
        rows.append({"feature": name, "mae_increase": float(np.mean(deltas))})

    return (pd.DataFrame(rows)
            .sort_values("mae_increase", ascending=False)
            .reset_index(drop=True))


def targets_table(metrics: dict) -> pd.DataFrame:
    """Build the brief's target-vs-achieved acceptance table.

    MAPE and R-squared are checked at the weekly aggregation level the brief
    specifies; MAE and RMSE are checked at daily SKU level, the only scale at
    which "<= 10 units" is a sensible threshold.

    Args:
        metrics: A dict with ``daily`` and ``weekly`` metric dicts.

    Returns:
        A DataFrame with metric, level, target, achieved and status columns.
    """
    checks = [
        ("MAPE (%)", "weekly", metrics["weekly"]["mape"], METRIC_TARGETS["mape"], "<="),
        ("R2", "weekly", metrics["weekly"]["r2"], METRIC_TARGETS["r2"], ">="),
        ("MAE (units)", "daily", metrics["daily"]["mae"], METRIC_TARGETS["mae"], "<="),
        ("RMSE (units)", "daily", metrics["daily"]["rmse"], METRIC_TARGETS["rmse"], "<="),
        ("MSE (units^2)", "daily", metrics["daily"]["mse"], float("nan"), ""),
    ]
    rows = []
    for name, level, achieved, target, op in checks:
        if op == "<=":
            status = "PASS" if achieved <= target else "FAIL"
        elif op == ">=":
            status = "PASS" if achieved >= target else "FAIL"
        else:
            status = "-"
        rows.append({"metric": name, "level": level, "target": target,
                     "achieved": round(achieved, 4), "status": status})
    return pd.DataFrame(rows)


def save_actual_vs_predicted_plot(joined: pd.DataFrame, out_path: str,
                                  n_points: int = 150) -> None:
    """Plot the LSTM's test-set predictions against actuals.

    Args:
        joined: DataFrame with ``y_true`` and ``y_pred`` columns.
        out_path: PNG path to write.
        n_points: Number of consecutive test points to display.

    Returns:
        None.
    """
    y_true = joined["y_true"].to_numpy()[:n_points]
    y_pred = joined["y_pred"].to_numpy()[:n_points]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_true, label="Actual", color="#333333", linewidth=1.4)
    ax.plot(y_pred, label="Predicted (LSTM)", color="#C44E52", alpha=0.85, linewidth=1.4)
    ax.set_title(f"Actual vs predicted units_sold - LSTM, test set (first {len(y_true)} rows)")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_loss_curves(histories: Dict[str, pd.DataFrame], out_path: str) -> None:
    """Plot train vs validation loss curves for every trained model.

    Args:
        histories: Mapping of model name -> per-epoch history DataFrame.
        out_path: PNG path to write.

    Returns:
        None.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history["epoch"], history["train_loss"], label="Train MSE")
        ax.plot(history["epoch"], history["val_loss"], label="Validation MSE")
        ax.set_title(f"{name}: training vs validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """CLI entry point: preprocess, train both models, score and save.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Train NeuralStock LSTM and MLP models")
    default_data = PROJECT_ROOT / "data" / "synthetic_inventory_demand.csv"
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--logs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for directory in (args.models_dir, args.logs_dir, args.outputs_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)

    pipeline = run_pipeline(args.data)
    train_df, val_df, test_df = pipeline["train_df"], pipeline["val_df"], pipeline["test_df"]
    feature_columns = pipeline["feature_columns"]
    print(f"Train {train_df.shape} | Val {val_df.shape} | Test {test_df.shape} "
          f"| {len(feature_columns)} features")

    with open(Path(args.models_dir) / "scaler.pkl", "wb") as handle:
        pickle.dump(pipeline["scaler"], handle)
    with open(Path(args.models_dir) / "feature_columns.json", "w") as handle:
        json.dump(feature_columns, handle, indent=2)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(args.logs_dir) / "neuralstock"))
    except Exception as exc:  # pragma: no cover - TensorBoard is optional
        print("TensorBoard unavailable, continuing without it:", exc)
        writer = None

    seq_len = args.sequence_length

    # ---------------- LSTM (sequence model) ----------------
    # Bridge the preceding split onto val/test so every evaluation row gets a
    # full window; without it the first `seq_len` rows of each SKU vanish and
    # the LSTM would be scored on a smaller sample than the MLP.
    val_bridged = bridge_for_windows(train_df, val_df, seq_len)
    seen_so_far = pd.concat([train_df, val_df], ignore_index=True)
    test_bridged = bridge_for_windows(seen_so_far, test_df, seq_len)

    train_seq = SlidingWindowDataset(train_df, feature_columns, sequence_length=seq_len)
    val_seq = SlidingWindowDataset(val_bridged, feature_columns, sequence_length=seq_len)
    test_seq = SlidingWindowDataset(test_bridged, feature_columns, sequence_length=seq_len)

    lstm_model, lstm_history = train_one_model(
        DemandLSTM(input_size=len(feature_columns)),
        DataLoader(train_seq, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_seq, batch_size=args.batch_size, shuffle=False),
        epochs=args.epochs, lr=args.lr, writer=writer, tag="LSTM", device=device,
    )
    torch.save(lstm_model.state_dict(), Path(args.models_dir) / "lstm_model.pt")

    lstm_test_pred = predict(lstm_model,
                             DataLoader(test_seq, batch_size=args.batch_size, shuffle=False),
                             device)
    lstm_train_pred = predict(lstm_model,
                              DataLoader(train_seq, batch_size=args.batch_size, shuffle=False),
                              device)
    lstm_metrics = score_daily_and_weekly(lstm_test_pred, test_bridged)
    lstm_train_metrics = score_daily_and_weekly(lstm_train_pred, train_df)

    # ---------------- MLP (tabular baseline) ----------------
    train_tab = TabularDataset(train_df, feature_columns)
    val_tab = TabularDataset(val_df, feature_columns)
    test_tab = TabularDataset(test_df, feature_columns)

    mlp_model, mlp_history = train_one_model(
        DemandMLP(input_size=len(feature_columns)),
        DataLoader(train_tab, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_tab, batch_size=args.batch_size, shuffle=False),
        epochs=args.epochs, lr=args.lr, writer=writer, tag="MLP", device=device,
    )
    torch.save(mlp_model.state_dict(), Path(args.models_dir) / "mlp_model.pt")

    mlp_test_pred = predict(mlp_model,
                            DataLoader(test_tab, batch_size=args.batch_size, shuffle=False),
                            device)
    mlp_metrics = score_daily_and_weekly(mlp_test_pred, test_df)

    # ---------------- Naive baseline ----------------
    naive_pred = lstm_test_pred.copy()
    naive_pred["y_pred"] = naive_pred["baseline"]
    naive_metrics = score_daily_and_weekly(naive_pred, test_bridged)

    # ---------------- Reporting ----------------
    print("\n=== Test-set metrics (daily SKU level) ===")
    for name, metrics in [("LSTM", lstm_metrics), ("MLP", mlp_metrics),
                          ("Naive (roll_mean_7)", naive_metrics)]:
        daily = metrics["daily"]
        print(f"{name:22s} MSE={daily['mse']:8.2f}  RMSE={daily['rmse']:6.2f}  "
              f"MAE={daily['mae']:6.2f}  MAPE={daily['mape']:6.2f}%  R2={daily['r2']:.3f}")

    print("\n=== Test-set metrics (weekly demand per category) ===")
    for name, metrics in [("LSTM", lstm_metrics), ("MLP", mlp_metrics),
                          ("Naive (roll_mean_7)", naive_metrics)]:
        weekly = metrics["weekly"]
        print(f"{name:22s} MSE={weekly['mse']:10.2f}  RMSE={weekly['rmse']:7.2f}  "
              f"MAE={weekly['mae']:7.2f}  MAPE={weekly['mape']:6.2f}%  R2={weekly['r2']:.3f}")

    overfit = overfitting_report(lstm_history, lstm_train_metrics["daily"],
                                 lstm_metrics["daily"])
    print(f"\nOverfitting check (LSTM): {overfit['verdict']} "
          f"(val/train loss ratio {overfit['val_over_train_loss_ratio']:.2f}, "
          f"test-minus-train MAPE {overfit['test_minus_train_mape_points']:.2f} pts)")

    print("\n=== Acceptance targets (LSTM) ===")
    acceptance = targets_table(lstm_metrics)
    print(acceptance.to_string(index=False))

    importance = permutation_importance(lstm_model, test_seq, feature_columns, device)
    importance.to_csv(Path(args.outputs_dir) / "feature_importance.csv", index=False)
    print("\nTop 8 features by permutation importance:")
    print(importance.head(8).to_string(index=False))

    with open(Path(args.outputs_dir) / "metrics.json", "w") as handle:
        json.dump(
            {
                "LSTM": lstm_metrics,
                "MLP": mlp_metrics,
                "Naive baseline": naive_metrics,
                "LSTM_train": lstm_train_metrics,
                "overfitting_check": overfit,
                "acceptance": acceptance.to_dict(orient="records"),
            },
            handle, indent=2,
        )

    lstm_history.to_csv(Path(args.outputs_dir) / "lstm_history.csv", index=False)
    mlp_history.to_csv(Path(args.outputs_dir) / "mlp_history.csv", index=False)
    save_loss_curves({"LSTM": lstm_history, "MLP": mlp_history},
                     str(Path(args.outputs_dir) / "loss_curves.png"))
    save_actual_vs_predicted_plot(
        lstm_test_pred.join(test_bridged[["date", "product_category"]], how="inner"),
        str(Path(args.outputs_dir) / "actual_vs_predicted.png"),
    )

    if writer is not None:
        writer.close()
    print(f"\nSaved models to {args.models_dir}/ and metrics/plots to {args.outputs_dir}/")


if __name__ == "__main__":
    main()
