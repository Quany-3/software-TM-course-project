"""Data utilities: sliding windows, normalization, and dataset classes."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def create_windows(data, window_size, stride=1):
    """
    Convert a time series into sliding windows.

    Args:
        data: np.ndarray of shape (n_timestamps, n_features) or (n_timestamps,)
        window_size: int, number of timestamps per window
        stride: int, step size between consecutive windows

    Returns:
        np.ndarray of shape (n_windows, window_size, n_features) or (n_windows, window_size)
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_timestamps, n_features = data.shape
    if n_timestamps < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} timestamps, got {n_timestamps}"
        )

    n_windows = (n_timestamps - window_size) // stride + 1

    windows = np.zeros((n_windows, window_size, n_features))
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start:start + window_size]

    if n_features == 1:
        return windows.squeeze(-1)  # (n_windows, window_size)
    return windows  # (n_windows, window_size, n_features)


def normalize(data, method='minmax'):
    """
    Normalize time series data.

    Args:
        data: np.ndarray of shape (n_timestamps, n_features)
        method: 'minmax' or 'zscore'

    Returns:
        normalized data, and (min, max) or (mean, std) for inverse transform
    """
    if method == 'minmax':
        d_min = data.min(axis=0, keepdims=True)
        d_max = data.max(axis=0, keepdims=True)
        denom = d_max - d_min
        denom[denom == 0] = 1.0
        normalized = (data - d_min) / denom
        return normalized, (d_min, d_max)

    elif method == 'zscore':
        d_mean = data.mean(axis=0, keepdims=True)
        d_std = data.std(axis=0, keepdims=True)
        d_std[d_std == 0] = 1.0
        normalized = (data - d_mean) / d_std
        return normalized, (d_mean, d_std)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_normalize(data, params, method='minmax'):
    """Apply saved normalization parameters to new data."""
    if method == 'minmax':
        d_min, d_max = params
        denom = d_max - d_min
        denom[denom == 0] = 1.0
        return (data - d_min) / denom
    elif method == 'zscore':
        d_mean, d_std = params
        d_std[d_std == 0] = 1.0
        return (data - d_mean) / d_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series sliding windows."""

    def __init__(self, windows):
        """
        Args:
            windows: np.ndarray of shape (n_windows, window_size) or
                     (n_windows, window_size, n_features)
        """
        self.windows = torch.FloatTensor(windows)
        if self.windows.dim() == 2:
            self.windows = self.windows.unsqueeze(-1)  # (N, W, 1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """Return flattened window for model input."""
        x = self.windows[idx]  # (window_size, n_features)
        return x.flatten()     # (window_size * n_features,)


class TimeSeriesDatasetDonut(Dataset):
    """PyTorch dataset for Donut (univariate, keeps 2D shape)."""

    def __init__(self, windows):
        """
        Args:
            windows: np.ndarray of shape (n_windows, window_size)
        """
        self.windows = torch.FloatTensor(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]  # (window_size,)


def prepare_data(data, window_size, stride=1, val_ratio=0.2, norm_method='minmax',
                 model_type='usad', random_seed=42):
    """
    Full data preparation pipeline.

    Args:
        data: np.ndarray, raw time series (n_timestamps,) or (n_timestamps, n_features)
        window_size: int
        stride: int
        val_ratio: float
        norm_method: 'minmax' or 'zscore'
        model_type: 'usad' or 'donut'
        random_seed: int

    Returns:
        train_loader, val_loader, norm_params, test_windows (for later scoring)
    """
    # Normalize
    data_2d = data if data.ndim == 2 else data.reshape(-1, 1)
    normalized, norm_params = normalize(data_2d, method=norm_method)

    # Create windows
    windows = create_windows(normalized, window_size, stride)

    # Create dataset
    if model_type == 'donut':
        dataset = TimeSeriesDatasetDonut(windows)
    else:
        dataset = TimeSeriesDataset(windows)

    # Split
    if len(dataset) < 2:
        raise ValueError(
            f"Need at least 2 windows for train/validation split, got {len(dataset)}"
        )

    n_val = int(len(dataset) * val_ratio)
    n_val = min(max(n_val, 1), len(dataset) - 1)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(random_seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # Keep all normalized windows for final scoring
    return train_loader, val_loader, norm_params, normalized, windows


def prepare_multivariate_data(data_dict, window_size, stride=1, val_ratio=0.2,
                              norm_method='minmax', random_seed=42):
    """
    Prepare multivariate data from a dict of univariate series.
    All series are aligned to the same time range and concatenated.

    Args:
        data_dict: dict of {name: np.ndarray(shape=(n_timestamps,))}
        window_size: int
        stride: int
        val_ratio: float
        norm_method: str
        random_seed: int

    Returns:
        train_loader, val_loader, norm_params, normalized, windows, feature_names
    """
    # Align lengths
    min_len = min(len(v) for v in data_dict.values())
    aligned = {k: v[:min_len] for k, v in data_dict.items()}
    feature_names = list(aligned.keys())

    # Stack into matrix
    data_matrix = np.column_stack([aligned[k] for k in feature_names])

    train_loader, val_loader, norm_params, normalized, windows = prepare_data(
        data_matrix, window_size, stride, val_ratio, norm_method,
        model_type='usad', random_seed=random_seed
    )

    return train_loader, val_loader, norm_params, normalized, windows, feature_names
