"""RL training pipeline that loads a frozen encoder and optimizes a portfolio policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from portfolio_rl.agent.policy.ppo_policy import PPOConfig, PPOPolicy
from portfolio_rl.agent.policy.sac_policy import SACConfig, SACPolicy
from portfolio_rl.agent.reward.composite_reward import CompositeReward
from portfolio_rl.data.loaders.registry import get_loader
from portfolio_rl.data.preprocessing.alignment import align_frames
from portfolio_rl.data.preprocessing.normalizer import RollingZScoreNormalizer
from portfolio_rl.encoding.asset_encoder.encoder_registry import EncoderRegistry
from portfolio_rl.environment.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from portfolio_rl.environment.risk_constraint import RiskConstraint
from portfolio_rl.environment.transaction_cost_model import TransactionCostModel
from portfolio_rl.fusion.attention_pooler import AttentionPooler
from portfolio_rl.fusion.state_builder import AssetSpec, StateBuilder
from portfolio_rl.training.monte_carlo import (
    MonteCarloEvaluationConfig,
    monte_carlo_evaluate,
    rollout_policy,
)


@dataclass
class TrainAgentConfig:
    """Configuration for RL training and out-of-sample evaluation."""

    asset_specs: Sequence[AssetSpec]
    encoder_checkpoint: str
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    algo: str = "ppo"
    window_size: int = 32
    normalizer_window: int = 20
    latent_dim: int = 16
    risk_scalar: float = 0.0
    sigma_max: float = 0.3
    transaction_cost_bps: float = 10.0
    feature_columns: Sequence[str] = ("open", "high", "low", "close", "volume")
    checkpoint_path: str = "artifacts/policy.pt"
    num_episodes: int = 20
    sac_warmup_steps: int = 256
    sac_updates_per_step: int = 1
    monte_carlo_num_simulations: int = 100
    monte_carlo_confidence_level: float = 0.95
    monte_carlo_seed: int = 42
    device: str = "cpu"


def run_training(config: TrainAgentConfig) -> Dict[str, object]:
    """Train a policy on the train split and evaluate it on the held-out split."""
    train_start = pd.Timestamp(config.train_start)
    train_end = pd.Timestamp(config.train_end)
    eval_start = pd.Timestamp(config.eval_start)
    eval_end = pd.Timestamp(config.eval_end)
    raw_frames = _load_price_frames(config.asset_specs, train_start, eval_end)
    aligned_raw = align_frames(raw_frames.values(), max_forward_fill=3)
    aligned_by_ticker = {asset.ticker: frame for asset, frame in zip(config.asset_specs, aligned_raw)}

    normalizers = _fit_normalizers(
        config.asset_specs,
        aligned_by_ticker,
        train_start,
        train_end,
        config.normalizer_window,
        config.feature_columns,
    )
    close_panel = _build_close_panel(config.asset_specs, aligned_by_ticker)
    encoder = EncoderRegistry(config.encoder_checkpoint, device=config.device).get_encoder()
    attention_pooler = AttentionPooler(config.latent_dim, num_heads=_choose_num_heads(config.latent_dim)).to(
        config.device
    )
    state_builder = StateBuilder(
        encoder=encoder,
        attention_pooler=attention_pooler,
        normalizers=normalizers,
        window_size=config.window_size,
        feature_columns=config.feature_columns,
        device=config.device,
        data_cache=aligned_by_ticker,
    )
    reward_fn = CompositeReward(
        lambda_vol=0.5,
        lambda_turn=0.1,
        lambda_hhi=0.05,
        window_size=20,
    )
    cost_model = TransactionCostModel(basis_points=config.transaction_cost_bps)
    risk_constraint = RiskConstraint(sigma_max=config.sigma_max)

    train_prices = close_panel.loc[(close_panel.index >= train_start) & (close_panel.index <= train_end)]
    eval_prices = close_panel.loc[(close_panel.index >= eval_start) & (close_panel.index <= eval_end)]
    train_env = PortfolioEnv(
        asset_specs=config.asset_specs,
        aligned_prices=train_prices,
        state_builder=state_builder,
        reward_fn=reward_fn,
        transaction_cost_model=cost_model,
        risk_constraint=risk_constraint,
        config=PortfolioEnvConfig(risk_scalar=config.risk_scalar),
    )
    eval_env = PortfolioEnv(
        asset_specs=config.asset_specs,
        aligned_prices=eval_prices,
        state_builder=state_builder,
        reward_fn=CompositeReward(
            lambda_vol=0.5,
            lambda_turn=0.1,
            lambda_hhi=0.05,
            window_size=20,
        ),
        transaction_cost_model=cost_model,
        risk_constraint=risk_constraint,
        config=PortfolioEnvConfig(risk_scalar=config.risk_scalar),
    )

    sample_state, _ = train_env.reset()
    policy = _build_policy(config, state_dim=sample_state.shape[0], num_assets=len(config.asset_specs))

    episode_logs: List[Dict[str, float]] = []
    for episode in range(config.num_episodes):
        if isinstance(policy, SACPolicy):
            episode_metrics = _run_sac_episode(policy, train_env, config)
        else:
            episode_metrics = _run_ppo_episode(policy, train_env)
        episode_metrics["episode"] = float(episode)
        episode_logs.append(episode_metrics)

    evaluation = evaluate_policy(policy, eval_env)
    monte_carlo_evaluation = monte_carlo_evaluate(
        policy,
        eval_env,
        MonteCarloEvaluationConfig(
            num_simulations=config.monte_carlo_num_simulations,
            confidence_level=config.monte_carlo_confidence_level,
            random_seed=config.monte_carlo_seed,
        ),
    )
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_path)

    return {
        "checkpoint_path": str(checkpoint_path),
        "episode_logs": episode_logs,
        "evaluation": evaluation,
        "monte_carlo_evaluation": monte_carlo_evaluation,
    }


def evaluate_policy(policy: PPOPolicy | SACPolicy, env: PortfolioEnv) -> Dict[str, float]:
    """Run a deterministic evaluation episode and compute out-of-sample Sharpe."""
    summary = rollout_policy(policy=policy, env=env, deterministic=True)
    return {
        "sharpe": summary["sharpe"],
        "total_return": summary["total_return"],
        "annualized_volatility": summary["annualized_volatility"],
        "max_drawdown": summary["max_drawdown"],
        "mean_turnover": summary["mean_turnover"],
        "final_portfolio_value": summary["final_portfolio_value"],
    }


def _run_sac_episode(policy: SACPolicy, env: PortfolioEnv, config: TrainAgentConfig) -> Dict[str, float]:
    state, _ = env.reset()
    done = False
    updates = 0
    losses = []
    returns = []

    while not done:
        if policy.replay_buffer.size < config.sac_warmup_steps:
            action = np.random.dirichlet(np.ones(len(env.asset_specs))).astype(np.float32)
        else:
            action = policy.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        policy.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        returns.append(info["gross_return"] - info["trading_cost"])

        if policy.replay_buffer.size >= policy.config.batch_size:
            for _ in range(config.sac_updates_per_step):
                batch = policy.replay_buffer.sample(policy.config.batch_size)
                losses.append(policy.update(batch))
                updates += 1

    return {
        "episode_return": float(env.portfolio_value / env.config.initial_cash - 1.0),
        "episode_sharpe": _annualized_sharpe(np.asarray(returns, dtype=np.float32)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "updates": float(updates),
    }


def _run_ppo_episode(policy: PPOPolicy, env: PortfolioEnv) -> Dict[str, float]:
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    returns = []

    while not done:
        action, log_prob, value = policy.act(state, deterministic=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value)
        log_probs.append(log_prob)
        returns.append(info["gross_return"] - info["trading_cost"])
        state = next_state

    gae = policy.compute_gae(
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_value=0.0,
    )
    loss = policy.update(
        {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "advantages": gae["advantages"],
            "returns": gae["returns"],
        }
    )
    return {
        "episode_return": float(env.portfolio_value / env.config.initial_cash - 1.0),
        "episode_sharpe": _annualized_sharpe(np.asarray(returns, dtype=np.float32)),
        "mean_loss": loss,
        "updates": 1.0,
    }


def _build_policy(config: TrainAgentConfig, state_dim: int, num_assets: int) -> PPOPolicy | SACPolicy:
    if config.algo.lower() == "sac":
        return SACPolicy(
            SACConfig(
                state_dim=state_dim,
                num_assets=num_assets,
                latent_dim=config.latent_dim,
                device=config.device,
            )
        )
    if config.algo.lower() == "ppo":
        return PPOPolicy(
            PPOConfig(
                state_dim=state_dim,
                num_assets=num_assets,
                latent_dim=config.latent_dim,
                device=config.device,
            )
        )
    raise ValueError("algo must be either 'ppo' or 'sac'")


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float(returns.mean()) if len(returns) == 1 else 0.0
    return float(np.sqrt(252.0) * returns.mean() / (returns.std(ddof=0) + 1e-8))


def _choose_num_heads(latent_dim: int) -> int:
    for heads in (4, 2, 1):
        if latent_dim % heads == 0:
            return heads
    return 1


def _fit_normalizers(
    asset_specs: Sequence[AssetSpec],
    aligned_by_ticker: Dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    window_size: int,
    feature_columns: Sequence[str],
) -> Dict[str, RollingZScoreNormalizer]:
    normalizers: Dict[str, RollingZScoreNormalizer] = {}
    for asset in asset_specs:
        frame = aligned_by_ticker[asset.ticker].loc[:, feature_columns]
        train_frame = frame.loc[(frame.index >= train_start) & (frame.index <= train_end)]
        normalizers[asset.ticker] = RollingZScoreNormalizer(window_size=window_size).fit(train_frame)
    return normalizers


def _build_close_panel(asset_specs: Sequence[AssetSpec], aligned_by_ticker: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = {asset.ticker: aligned_by_ticker[asset.ticker]["close"] for asset in asset_specs}
    return pd.DataFrame(data).sort_index()


def _load_price_frames(
    asset_specs: Sequence[AssetSpec],
    train_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    fetch_start = (train_start - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    fetch_end = (eval_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames: Dict[str, pd.DataFrame] = {}
    for asset in asset_specs:
        loader = get_loader(asset.asset_type)
        frames[asset.ticker] = loader.fetch(asset.ticker, fetch_start, fetch_end)
    return frames
