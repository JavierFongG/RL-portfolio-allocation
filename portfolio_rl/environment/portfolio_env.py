"""Gymnasium environment for daily portfolio rebalancing with learned states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from portfolio_rl.agent.reward.base_reward import BaseReward
from portfolio_rl.environment.risk_constraint import RiskConstraint, project_to_simplex
from portfolio_rl.environment.transaction_cost_model import TransactionCostModel
from portfolio_rl.fusion.state_builder import AssetSpec, StateBuilder


@dataclass
class PortfolioEnvConfig:
    """Configuration for the portfolio environment rollout dynamics."""

    initial_cash: float = 1.0
    risk_scalar: float = 0.0


class PortfolioEnv(gym.Env):
    """Environment that simulates daily rebalancing over aligned asset returns."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        asset_specs: Sequence[AssetSpec],
        aligned_prices: pd.DataFrame,
        state_builder: StateBuilder,
        reward_fn: BaseReward,
        transaction_cost_model: TransactionCostModel,
        risk_constraint: RiskConstraint | None = None,
        config: PortfolioEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.asset_specs = list(asset_specs)
        self.asset_names = [asset.ticker for asset in self.asset_specs]
        self.aligned_prices = aligned_prices.loc[:, self.asset_names].sort_index()
        self.returns = self.aligned_prices.pct_change().fillna(0.0)
        self.state_builder = state_builder
        self.reward_fn = reward_fn
        self.transaction_cost_model = transaction_cost_model
        self.risk_constraint = risk_constraint
        self.config = config or PortfolioEnvConfig()

        self.num_assets = len(self.asset_specs)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        sample_state = self.state_builder.build_state(
            self.aligned_prices.index[0], self.asset_specs, self.config.risk_scalar
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_state.shape,
            dtype=np.float32,
        )

        self.current_step = 0
        self.portfolio_value = self.config.initial_cash
        self.prev_weights = np.full(self.num_assets, 1.0 / self.num_assets, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.config.initial_cash
        self.prev_weights = np.full(self.num_assets, 1.0 / self.num_assets, dtype=np.float32)
        self.reward_fn.reset()
        state = self.state_builder.build_state(
            self.aligned_prices.index[self.current_step], self.asset_specs, self.config.risk_scalar
        )
        return state, {"portfolio_value": self.portfolio_value, "turnover": 0.0}

    def step(self, weights: np.ndarray):
        weights = np.asarray(weights, dtype=np.float32)
        weights = project_to_simplex(weights)

        covariance = self.returns.iloc[max(0, self.current_step - 19) : self.current_step + 1].cov().to_numpy()
        if self.risk_constraint is not None and covariance.size > 0 and not np.isnan(covariance).any():
            weights = self.risk_constraint.project(weights, covariance)

        next_step = self.current_step + 1
        terminated = next_step >= len(self.aligned_prices.index) - 1
        truncated = False

        returns_vector = self.returns.iloc[next_step].to_numpy(dtype=np.float32)
        gross_return = float(np.dot(weights, returns_vector))
        turnover = float(np.abs(weights - self.prev_weights).sum())
        trading_cost = self.transaction_cost_model.compute(weights, self.prev_weights)
        reward_output = self.reward_fn.compute(weights, returns_vector, self.prev_weights)
        reward_terms = {}
        if isinstance(reward_output, tuple):
            reward, reward_terms = reward_output
        else:
            reward = reward_output

        self.portfolio_value *= 1.0 + gross_return - trading_cost
        self.prev_weights = weights.astype(np.float32)
        self.current_step = next_step

        next_state = self.state_builder.build_state(
            self.aligned_prices.index[self.current_step], self.asset_specs, self.config.risk_scalar
        )
        info = {
            "portfolio_value": float(self.portfolio_value),
            "turnover": turnover,
            "gross_return": gross_return,
            "trading_cost": trading_cost,
            "reward_terms": reward_terms,
        }
        return next_state, float(reward), terminated, truncated, info
