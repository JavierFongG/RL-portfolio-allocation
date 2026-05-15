"""Reward that penalizes tail risk through a rolling CVaR estimate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from portfolio_rl.agent.reward.base_reward import BaseReward


@dataclass
class CVaRReward(BaseReward):
    """Combines realized return with a conditional value-at-risk penalty."""

    risk_aversion: float = 1.0
    confidence_level: float = 0.95
    lookback: int = 60
    _history: deque = field(default_factory=deque, init=False, repr=False)

    def reset(self) -> None:
        self._history.clear()

    def compute(self, weights: np.ndarray, returns: np.ndarray, prev_weights: np.ndarray) -> float:
        portfolio_return = float(np.dot(weights, returns))
        self._history.append(portfolio_return)
        while len(self._history) > self.lookback:
            self._history.popleft()

        series = np.asarray(self._history, dtype=np.float32)
        if len(series) == 0:
            return portfolio_return

        losses = -series
        quantile = float(np.quantile(losses, self.confidence_level))
        tail_losses = losses[losses >= quantile]
        cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else 0.0
        return portfolio_return - self.risk_aversion * cvar
