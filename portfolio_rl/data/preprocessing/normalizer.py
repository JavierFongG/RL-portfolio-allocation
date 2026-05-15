"""Train-only rolling z-score normalization for per-asset market features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RollingZScoreNormalizer:
    """Computes z-scores using only historical observations available up to each timestamp."""

    window_size: int
    min_periods: int = 5
    epsilon: float = 1e-8
    _train_history: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _feature_means: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _feature_stds: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "RollingZScoreNormalizer":
        """Store training history and summary statistics for future transforms."""
        if df.empty:
            raise ValueError("Cannot fit normalizer on an empty DataFrame")

        frame = self._validate_frame(df)
        self._train_history = frame.tail(max(self.window_size - 1, 1)).copy()
        self._feature_means = frame.mean().to_dict()
        stds = frame.std(ddof=0).replace(0.0, 1.0)
        self._feature_stds = stds.to_dict()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data without refitting, using trailing training context where needed."""
        if not self._fitted or self._train_history is None:
            raise RuntimeError("Normalizer must be fitted before calling transform")

        frame = self._validate_frame(df)
        history = self._train_history
        combined = pd.concat([history, frame], axis=0)
        rolling_mean = combined.rolling(self.window_size, min_periods=self.min_periods).mean().shift(1)
        rolling_std = combined.rolling(self.window_size, min_periods=self.min_periods).std(ddof=0).shift(1)

        fallback_mean = pd.Series(self._feature_means)
        fallback_std = pd.Series(self._feature_stds).replace(0.0, 1.0)
        rolling_mean = rolling_mean.fillna(fallback_mean)
        rolling_std = rolling_std.replace(0.0, np.nan).fillna(fallback_std)

        normalized = (combined - rolling_mean) / (rolling_std + self.epsilon)
        normalized = normalized.loc[frame.index]
        normalized = normalized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return normalized.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the provided training data and return its normalized representation."""
        return self.fit(df).transform(df)

    def _validate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Expected a DataFrame indexed by pandas.DatetimeIndex")
        frame = df.sort_index().astype(float)
        if frame.isnull().all(axis=None):
            raise ValueError("Input frame contains only missing values")
        return frame

