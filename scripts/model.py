"""model.py

PyTorch architectures and Dataset / DataLoader utilities for NeuralStock.

Two models:
    * :class:`DemandLSTM` - a stacked LSTM over a sliding window of past
      daily feature rows for one SKU (sequence forecasting).
    * :class:`DemandMLP` - a feedforward baseline over a single engineered
      feature row, used to show what the sequence model buys us.

Both datasets return the same ``(X, y, baseline, row_index)`` tuple shape, so
the same training and evaluation code drives either one, and predictions can
always be joined back to their ``date`` / ``product_category`` for the weekly
aggregation the brief scores on.
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
    """Set python, numpy and torch seeds for reproducibility.

    Args:
        seed: Seed value applied everywhere.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SlidingWindowDataset(Dataset):
    """Fixed-length sliding windows of feature rows, per SKU.

    Each sample is a ``(sequence_length, n_features)`` window of consecutive
    daily rows for one SKU, paired with the target of the row immediately
    after the window. Windows never cross a ``product_id`` boundary.

    Residual targets
    ----------------
    With ``predict_residual=True`` the label is ``units_sold - roll_mean_7_raw``
    rather than raw units. Daily demand has a large mean relative to its
    day-to-day variation, so training on raw units lets the network minimise
    MSE almost entirely by parking its output bias at the series mean and
    contributing nothing from the recurrent weights. The residual is centred
    near zero whatever a SKU's level, which keeps gradients flowing into the
    sequence weights. Add ``baseline`` back onto the output to recover a
    real-units forecast.

    Args:
        df: Scaled, feature-engineered DataFrame sorted by product_id, date.
            ``target_column`` and ``baseline_column`` must remain unscaled.
        feature_columns: Column names used as model inputs.
        target_column: Name of the raw target column.
        baseline_column: Unscaled rolling-mean column used as the residual
            baseline and as the naive comparator.
        sequence_length: Number of past daily steps per window.
        predict_residual: If True, the label is ``target - baseline``.
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

        windows, labels, baselines, indices = [], [], [], []
        for _, group in df.groupby("product_id", sort=False):
            group = group.sort_values("date")
            row_ids = group.index.to_numpy()
            features = group[feature_columns].to_numpy(dtype=np.float32)
            targets = group[target_column].to_numpy(dtype=np.float32)
            base = group[baseline_column].to_numpy(dtype=np.float32)

            for i in range(len(group) - sequence_length):
                j = i + sequence_length
                windows.append(features[i:j])
                labels.append(targets[j] - base[j] if predict_residual else targets[j])
                baselines.append(base[j])
                indices.append(row_ids[j])

        self.X = np.asarray(windows, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.float32)
        self.baseline = np.asarray(baselines, dtype=np.float32)
        self.row_index = np.asarray(indices)

    def __len__(self) -> int:
        """Return the number of sliding-window samples.

        Returns:
            Total sample count across all SKUs.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """Fetch one ``(window, label, baseline, row_index)`` sample.

        Args:
            idx: Sample index.

        Returns:
            A tuple of the window tensor ``[sequence_length, n_features]``,
            the scalar label, the scalar real-units baseline, and the source
            DataFrame row index for joining predictions back to date/category.
        """
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.baseline[idx], dtype=torch.float32),
            torch.tensor(int(self.row_index[idx]), dtype=torch.long),
        )


class TabularDataset(Dataset):
    """One engineered feature row per sample, for the MLP baseline.

    Uses the identical target formulation as :class:`SlidingWindowDataset` so
    the two models are compared on the same quantity - the original pipeline
    trained the MLP on raw units and the LSTM on residuals, which made the
    reported comparison meaningless.

    Args:
        df: Scaled, feature-engineered DataFrame.
        feature_columns: Column names used as model inputs.
        target_column: Name of the raw target column.
        baseline_column: Unscaled rolling-mean baseline column.
        predict_residual: If True, the label is ``target - baseline``.
    """

    def __init__(self, df: pd.DataFrame, feature_columns: List[str],
                 target_column: str = "units_sold",
                 baseline_column: str = "roll_mean_7_raw",
                 predict_residual: bool = True) -> None:
        self.X = df[feature_columns].to_numpy(dtype=np.float32)
        raw = df[target_column].to_numpy(dtype=np.float32)
        self.baseline = df[baseline_column].to_numpy(dtype=np.float32)
        self.y = (raw - self.baseline) if predict_residual else raw
        self.row_index = df.index.to_numpy()

    def __len__(self) -> int:
        """Return the number of rows.

        Returns:
            Row count.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """Fetch one ``(features, label, baseline, row_index)`` sample.

        Args:
            idx: Row index.

        Returns:
            A tuple of the feature tensor, scalar label, scalar baseline and
            source DataFrame row index.
        """
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.baseline[idx], dtype=torch.float32),
            torch.tensor(int(self.row_index[idx]), dtype=torch.long),
        )


class DemandLSTM(nn.Module):
    """Stacked LSTM for sequence-based daily demand forecasting.

    Architecture: 2-layer LSTM (hidden 96) -> dropout -> Linear(96, 32) ->
    ReLU -> Linear(32, 1). Two layers and a modest hidden size are enough
    here: each SKU contributes ~700 daily observations, and a wider network
    starts memorising the training period within a handful of epochs.

    Args:
        input_size: Number of features per time step.
        hidden_size: LSTM hidden-state width.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers and before the head.
    """

    def __init__(self, input_size: int, hidden_size: int = 96,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``[batch, sequence_length, input_size]``.

        Returns:
            Tensor of shape ``[batch]`` with the predicted value (a residual
            when the dataset was built with ``predict_residual=True``).
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])
        return self.head(last_hidden).squeeze(-1)


class DemandMLP(nn.Module):
    """Feedforward baseline over a single engineered feature row.

    Architecture: Linear(128) -> ReLU -> Dropout -> Linear(64) -> ReLU ->
    Dropout -> Linear(32) -> ReLU -> Linear(1).

    Args:
        input_size: Number of input features.
        dropout: Dropout probability after the first two hidden layers.
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
            x: Input tensor of shape ``[batch, input_size]``.

        Returns:
            Tensor of shape ``[batch]`` with the predicted value.
        """
        return self.net(x).squeeze(-1)
