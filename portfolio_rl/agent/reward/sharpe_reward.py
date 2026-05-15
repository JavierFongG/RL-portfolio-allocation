"""Reward based on a rolling estimate of the portfolio Sharpe ratio."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from portfolio_rl.agent.reward.base_reward import BaseReward


@dataclass
class SharpeReward(BaseReward):
    """Computes a rolling Sharpe ratio from realized portfolio returns."""

    lookback: int = 20
    risk_free_rate: float = 0.0
    annualization_factor: float = 252.0
    epsilon: float = 1e-8
    _history: deque = field(default_factory=deque, init=False, repr=False)

    def reset(self) -> None:
        self._history.clear()

    def compute(self, weights: np.ndarray, returns: np.ndarray, prev_weights: np.ndarray) -> float:
        portfolio_return = float(np.dot(weights, returns) - self.risk_free_rate / self.annualization_factor)
        self._history.append(portfolio_return)
        while len(self._history) > self.lookback:
            self._history.popleft()

        series = np.asarray(self._history, dtype=np.float32)
        if len(series) < 2:
            return portfolio_return
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        sharpe = np.sqrt(self.annualization_factor) * mean / (std + self.epsilon)
        return sharpe
