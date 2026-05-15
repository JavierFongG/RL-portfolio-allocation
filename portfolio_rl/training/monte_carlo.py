"""Monte Carlo evaluation utilities for robustness testing of trained portfolio policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from portfolio_rl.agent.policy.ppo_policy import PPOPolicy
from portfolio_rl.agent.policy.sac_policy import SACPolicy
from portfolio_rl.environment.portfolio_env import PortfolioEnv


@dataclass
class MonteCarloEvaluationConfig:
    """Configuration for repeated stochastic policy rollouts on the evaluation environment."""

    num_simulations: int = 100
    confidence_level: float = 0.95
    random_seed: int = 42


def monte_carlo_evaluate(
    policy: PPOPolicy | SACPolicy,
    env: PortfolioEnv,
    config: MonteCarloEvaluationConfig,
) -> Dict[str, object]:
    """Run repeated stochastic rollouts and summarize the distribution of outcomes."""
    if config.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    rng = np.random.default_rng(config.random_seed)
    simulation_summaries: List[Dict[str, float]] = []

    for simulation_id in range(config.num_simulations):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        np.random.seed(seed)
        torch.manual_seed(seed)
        summary = rollout_policy(policy=policy, env=env, deterministic=False)
        summary["simulation"] = float(simulation_id)
        summary["seed"] = float(seed)
        simulation_summaries.append(summary)

    alpha = 1.0 - config.confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    sharpe_values = np.asarray([item["sharpe"] for item in simulation_summaries], dtype=np.float64)
    return_values = np.asarray([item["total_return"] for item in simulation_summaries], dtype=np.float64)
    drawdown_values = np.asarray([item["max_drawdown"] for item in simulation_summaries], dtype=np.float64)
    turnover_values = np.asarray([item["mean_turnover"] for item in simulation_summaries], dtype=np.float64)
    final_values = np.asarray([item["final_portfolio_value"] for item in simulation_summaries], dtype=np.float64)

    return {
        "num_simulations": config.num_simulations,
        "confidence_level": config.confidence_level,
        "random_seed": config.random_seed,
        "summary": {
            "sharpe_mean": float(sharpe_values.mean()),
            "sharpe_std": float(sharpe_values.std(ddof=0)),
            "sharpe_median": float(np.median(sharpe_values)),
            "sharpe_ci_lower": float(np.percentile(sharpe_values, lower_q)),
            "sharpe_ci_upper": float(np.percentile(sharpe_values, upper_q)),
            "return_mean": float(return_values.mean()),
            "return_std": float(return_values.std(ddof=0)),
            "return_median": float(np.median(return_values)),
            "return_ci_lower": float(np.percentile(return_values, lower_q)),
            "return_ci_upper": float(np.percentile(return_values, upper_q)),
            "max_drawdown_mean": float(drawdown_values.mean()),
            "max_drawdown_median": float(np.median(drawdown_values)),
            "turnover_mean": float(turnover_values.mean()),
            "turnover_median": float(np.median(turnover_values)),
            "final_value_mean": float(final_values.mean()),
            "final_value_median": float(np.median(final_values)),
            "probability_of_loss": float(np.mean(return_values < 0.0)),
        },
        "simulations": simulation_summaries,
    }


def rollout_policy(
    policy: PPOPolicy | SACPolicy,
    env: PortfolioEnv,
    deterministic: bool,
) -> Dict[str, float]:
    """Execute one rollout and compute performance diagnostics."""
    state, _ = env.reset()
    done = False
    daily_returns: List[float] = []
    turnovers: List[float] = []
    portfolio_values: List[float] = [float(env.portfolio_value)]

    while not done:
        action, _, _ = policy.act(state, deterministic=deterministic)
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        daily_returns.append(info["gross_return"] - info["trading_cost"])
        turnovers.append(info["turnover"])
        portfolio_values.append(info["portfolio_value"])

    returns_array = np.asarray(daily_returns, dtype=np.float64)
    turnover_array = np.asarray(turnovers, dtype=np.float64)
    value_array = np.asarray(portfolio_values, dtype=np.float64)
    return {
        "sharpe": _annualized_sharpe(returns_array),
        "total_return": float(env.portfolio_value / env.config.initial_cash - 1.0),
        "annualized_volatility": _annualized_volatility(returns_array),
        "max_drawdown": _max_drawdown(value_array),
        "mean_turnover": float(turnover_array.mean()) if len(turnover_array) else 0.0,
        "final_portfolio_value": float(env.portfolio_value),
    }


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float(returns.mean()) if len(returns) == 1 else 0.0
    return float(np.sqrt(252.0) * returns.mean() / (returns.std(ddof=0) + 1e-8))


def _annualized_volatility(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.sqrt(252.0) * returns.std(ddof=0))


def _max_drawdown(portfolio_values: np.ndarray) -> float:
    if len(portfolio_values) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(portfolio_values)
    drawdowns = 1.0 - (portfolio_values / np.maximum(running_peak, 1e-8))
    return float(drawdowns.max())
