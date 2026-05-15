"""Utilities for turning tabular time series into sliding window tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_sequences(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """Create overlapping fixed-length windows from a feature frame."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(df) < window_size:
        raise ValueError("DataFrame is shorter than the requested window size")

    values = df.to_numpy(dtype=np.float32, copy=True)
    sequences = [values[start : start + window_size] for start in range(len(values) - window_size + 1)]
    return np.stack(sequences, axis=0)

