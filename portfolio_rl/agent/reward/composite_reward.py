"""Composite reward implementing normalized return, volatility, turnover, and concentration terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio_rl.agent.reward.base_reward import BaseReward


@dataclass
class CompositeReward(BaseReward):
    """Computes normalized portfolio reward terms with EMA-based scale normalization."""

    lambda_vol: float
    lambda_turn: float
    lambda_hhi: float
    window_size: int
    ema_decay: float = 0.99
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        self._log_return_history: list[float] = []
        self._scale_ema = {
            "r_p": 1.0,
            "sigma_p": 1.0,
            "turnover": 1.0,
            "hhi": 1.0,
        }

    def reset(self) -> None:
        self._log_return_history.clear()
        for key in self._scale_ema:
            self._scale_ema[key] = 1.0

    def compute(self, weights: np.ndarray, returns: np.ndarray, prev_weights: np.ndarray) -> tuple[float, dict]:
        weights = np.asarray(weights, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        prev_weights = np.asarray(prev_weights, dtype=np.float64)
        if weights.ndim != 1 or returns.ndim != 1 or prev_weights.ndim != 1:
            raise ValueError("weights, returns, and prev_weights must be 1D arrays")
        if not (len(weights) == len(returns) == len(prev_weights)):
            raise ValueError("weights, returns, and prev_weights must have the same length")

        portfolio_simple_return = float(np.dot(weights, returns))
        if portfolio_simple_return <= -1.0:
            raise ValueError("Portfolio return must be greater than -1 to compute log return")
        r_p = float(np.log1p(portfolio_simple_return))

        self._log_return_history.append(r_p)
        if len(self._log_return_history) > self.window_size:
            self._log_return_history.pop(0)
        sigma_p = float(np.std(self._log_return_history, ddof=0))

        turnover = float(np.abs(weights - prev_weights).sum())
        hhi = float(np.sum(np.square(weights)))
        n_assets = len(weights)
        lower_bound = 1.0 / n_assets
        if not (lower_bound - self.epsilon <= hhi <= 1.0 + self.epsilon):
            raise AssertionError(f"HHI={hhi} is outside the expected range [{lower_bound}, 1]")

        raw_terms = {
            "r_p": r_p,
            "sigma_p": sigma_p,
            "turnover": turnover,
            "hhi": hhi,
        }
        normalized_terms = {
            name: self._normalize_term(name, value) for name, value in raw_terms.items()
        }
        reward = (
            normalized_terms["r_p"]
            - self.lambda_vol * normalized_terms["sigma_p"]
            - self.lambda_turn * normalized_terms["turnover"]
            - self.lambda_hhi * normalized_terms["hhi"]
        )
        terms = {
            "reward": float(reward),
            "raw_r_p": raw_terms["r_p"],
            "raw_sigma_p": raw_terms["sigma_p"],
            "raw_turnover": raw_terms["turnover"],
            "raw_hhi": raw_terms["hhi"],
            "norm_r_p": normalized_terms["r_p"],
            "norm_sigma_p": normalized_terms["sigma_p"],
            "norm_turnover": normalized_terms["turnover"],
            "norm_hhi": normalized_terms["hhi"],
            "scale_r_p": self._scale_ema["r_p"],
            "scale_sigma_p": self._scale_ema["sigma_p"],
            "scale_turnover": self._scale_ema["turnover"],
            "scale_hhi": self._scale_ema["hhi"],
        }
        return float(reward), terms

    def _normalize_term(self, name: str, value: float) -> float:
        updated_scale = self.ema_decay * self._scale_ema[name] + (1.0 - self.ema_decay) * abs(value)
        self._scale_ema[name] = max(updated_scale, self.epsilon)
        return float(value / self._scale_ema[name])
