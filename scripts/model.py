"""
model.py

PyTorch model architectures and Dataset/DataLoader utilities for
NeuralStock demand forecasting.

Two models are provided:
    - DemandLSTM: a stacked LSTM that consumes a sliding window of past
      observations per product (sequence forecasting).
    - DemandMLP: a feedforward baseline that consumes a single row of
      engineered features (no explicit sequence modelling) — used to show
      the LSTM's improvement over a simpler architecture.

Requires: torch >= 2.0 (install via requirements.txt: `pip install torch`).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set python, numpy, and torch random seeds for reproducibility.

    Args:
        seed: The seed value to apply everywhere.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SlidingWindowDataset(Dataset):
    """Builds fixed-length sliding windows of features per product_id.

    Each sample is a (sequence_length, n_features) window of past feature
    rows for a single SKU, paired with a target for the row immediately
    following the window, and a baseline value for that same row. Windows
    never cross product_id boundaries.

    By default (predict_residual=True) the target handed to the model is
    `units_sold - roll_mean_7_raw` rather than raw units_sold. units_sold
    has a much larger mean than variance across the series (~28 vs a
    typical day-to-day swing of a few units), so training directly on it
    lets an LSTM minimize MSE almost entirely by pushing its output bias
    to the global mean while contributing near-zero from the recurrent
    weights — the model "solves" training by memorizing a constant. The
    residual is centered near zero regardless of a SKU's overall level,
    which keeps gradients meaningful for the actual sequence weights. Add
    the returned baseline back onto the model's output to recover a
    real-units forecast (see train.py's evaluate()).

    Args:
        df: Feature-engineered, scaled DataFrame sorted by product_id, date.
            Must contain target_column and baseline_column UNSCALED even
            if feature_columns are scaled (apply_scaler in preprocess.py
            leaves both alone by design).
        feature_columns: List of column names to use as model inputs.
        target_column: Name of the raw target column.
        baseline_column: Name of the unscaled rolling-mean column used as
            the residual baseline and as the naive-forecast comparator.
        sequence_length: Number of past time steps per window.
        predict_residual: If True (default), the label returned is
            `target - baseline` instead of the raw target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "units_sold",
        baseline_column: str = "roll_mean_7_raw",
        sequence_length: int = 14,
        predict_residual: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.predict_residual = predict_residual
        self.samples: List[Tuple[np.ndarray, float, float]] = []

        for _, group in df.groupby("product_id"):
            group = group.reset_index(drop=True)
            features = group[feature_columns].to_numpy(dtype=np.float32)
            targets = group[target_column].to_numpy(dtype=np.float32)
            baselines = group[baseline_column].to_numpy(dtype=np.float32)

            for i in range(len(group) - sequence_length):
                window = features[i : i + sequence_length]
                target = targets[i + sequence_length]
                baseline = baselines[i + sequence_length]
                label = (target - baseline) if predict_residual else target
                self.samples.append((window, label, baseline))

    def __len__(self) -> int:
        """Return the number of sliding-window samples.

        Returns:
            Total sample count across all products.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one (window, target, baseline) triple as tensors.

        Args:
            idx: Sample index.

        Returns:
            A tuple of (window tensor of shape [sequence_length, n_features],
            scalar target tensor, scalar baseline tensor). The baseline is
            the row's raw roll_mean_7 value regardless of predict_residual,
            so callers can always reconstruct real-units predictions/targets
            and compute the roll_mean_7 naive-forecast comparison.
        """
        window, label, baseline = self.samples[idx]
        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(baseline, dtype=torch.float32),
        )


class TabularDataset(Dataset):
    """Wraps a single row of engineered features per sample (for the MLP baseline).

    Unlike SlidingWindowDataset, the MLP is trained directly on the raw
    target — its short, largely linear ReLU path from the lag/rolling
    features to the output has no trouble learning the series' overall
    level, so it doesn't need the residual trick. A baseline value is
    still returned (unused in the MLP's own loss) purely so train.py can
    report the roll_mean_7 naive-forecast comparison against the same
    DataLoader, using the same tuple shape as SlidingWindowDataset.

    Args:
        df: Feature-engineered, scaled DataFrame.
        feature_columns: List of column names to use as model inputs.
        target_column: Name of the target column (kept raw/unscaled).
        baseline_column: Name of the unscaled rolling-mean column used
            only for the naive-forecast comparison in train.py.
    """

    def __init__(self, df: pd.DataFrame, feature_columns: List[str],
                 target_column: str = "units_sold",
                 baseline_column: str = "roll_mean_7_raw") -> None:
        self.X = df[feature_columns].to_numpy(dtype=np.float32)
        self.y = df[target_column].to_numpy(dtype=np.float32)
        self.baseline = df[baseline_column].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        """Return the number of rows.

        Returns:
            Row count.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one (features, target, baseline) triple as tensors.

        Args:
            idx: Row index.

        Returns:
            A tuple of (feature tensor, scalar raw-target tensor, scalar
            baseline tensor).
        """
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.baseline[idx], dtype=torch.float32),
        )


class DemandLSTM(nn.Module):
    """Stacked LSTM for sequence-based weekly demand forecasting.

    Architecture: 2-layer LSTM -> dropout -> linear head producing a single
    scalar forecast. Hidden size and layer count are kept modest (64, 2)
    since each SKU only has ~100-150 historical observations — a larger
    network would overfit quickly on a sequence this short.

    Args:
        input_size: Number of features per time step.
        hidden_size: LSTM hidden state dimensionality.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers and before the head.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape [batch, sequence_length, input_size].

        Returns:
            Tensor of shape [batch] with the predicted units_sold.
        """
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # final layer's last hidden state: [batch, hidden]
        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden).squeeze(-1)


class DemandMLP(nn.Module):
    """Feedforward baseline model operating on a single engineered feature row.

    Architecture: 3 hidden layers (128 -> 64 -> 32) with ReLU activations
    and dropout, ending in a single-unit regression head.

    Args:
        input_size: Number of input features.
        dropout: Dropout probability after each hidden layer.
    """

    def __init__(self, input_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape [batch, input_size].

        Returns:
            Tensor of shape [batch] with the predicted units_sold.
        """
        return self.net(x).squeeze(-1)
