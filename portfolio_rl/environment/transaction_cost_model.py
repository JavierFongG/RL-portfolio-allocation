"""Transaction cost model based on turnover measured by L1 weight changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransactionCostModel:
    """Computes linear transaction costs from portfolio turnover."""

    basis_points: float = 10.0

    def compute(self, w_new: np.ndarray, w_old: np.ndarray) -> float:
        rate = self.basis_points / 10_000.0
        return float(rate * np.abs(w_new - w_old).sum())

