"""Walk-forward orchestration for sequential train, validation, and test portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from portfolio_rl.fusion.state_builder import AssetSpec
from portfolio_rl.training.pretrain_encoder import PretrainEncoderConfig, run_pretraining
from portfolio_rl.training.train_agent import TrainAgentConfig, run_training


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""

    asset_specs: Sequence[AssetSpec]
    start_date: str
    end_date: str
    train_days: int
    val_days: int
    test_days: int
    purge_gap_days: int
    output_dir: str = "artifacts/walk_forward"
    step_days: int | None = None
    window_size: int = 32
    hidden_dim: int = 64
    latent_dim: int = 16
    algo: str = "ppo"
    normalizer_window: int = 20
    num_episodes: int = 20
    monte_carlo_num_simulations: int = 100
    monte_carlo_confidence_level: float = 0.95
    monte_carlo_seed: int = 42
    device: str = "cpu"


def run_walk_forward(config: WalkForwardConfig) -> List[Dict[str, object]]:
    """Run walk-forward training across sequential folds and report test Sharpe values."""
    folds = make_walk_forward_splits(config)
    results: List[Dict[str, object]] = []

    for fold_id, fold in enumerate(folds):
        fold_dir = Path(config.output_dir) / f"fold_{fold_id:02d}"
        encoder_path = fold_dir / "encoder.pt"
        policy_path = fold_dir / "policy.pt"

        run_pretraining(
            PretrainEncoderConfig(
                asset_specs=config.asset_specs,
                train_start=fold["train_start"].strftime("%Y-%m-%d"),
                train_end=fold["train_end"].strftime("%Y-%m-%d"),
                val_start=fold["val_start"].strftime("%Y-%m-%d"),
                val_end=fold["val_end"].strftime("%Y-%m-%d"),
                window_size=config.window_size,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                normalizer_window=config.normalizer_window,
                checkpoint_path=str(encoder_path),
            )
        )
        training_result = run_training(
            TrainAgentConfig(
                asset_specs=config.asset_specs,
                encoder_checkpoint=str(encoder_path),
                train_start=fold["train_start"].strftime("%Y-%m-%d"),
                train_end=fold["train_end"].strftime("%Y-%m-%d"),
                eval_start=fold["test_start"].strftime("%Y-%m-%d"),
                eval_end=fold["test_end"].strftime("%Y-%m-%d"),
                algo=config.algo,
                window_size=config.window_size,
                latent_dim=config.latent_dim,
                normalizer_window=config.normalizer_window,
                checkpoint_path=str(policy_path),
                num_episodes=config.num_episodes,
                monte_carlo_num_simulations=config.monte_carlo_num_simulations,
                monte_carlo_confidence_level=config.monte_carlo_confidence_level,
                monte_carlo_seed=config.monte_carlo_seed,
                device=config.device,
            )
        )
        mc_summary = training_result["monte_carlo_evaluation"]["summary"]
        results.append(
            {
                "fold": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "out_of_sample_sharpe": training_result["evaluation"]["sharpe"],
                "out_of_sample_return": training_result["evaluation"]["total_return"],
                "monte_carlo_sharpe_mean": mc_summary["sharpe_mean"],
                "monte_carlo_sharpe_ci_lower": mc_summary["sharpe_ci_lower"],
                "monte_carlo_sharpe_ci_upper": mc_summary["sharpe_ci_upper"],
                "monte_carlo_return_mean": mc_summary["return_mean"],
                "monte_carlo_return_ci_lower": mc_summary["return_ci_lower"],
                "monte_carlo_return_ci_upper": mc_summary["return_ci_upper"],
                "monte_carlo_probability_of_loss": mc_summary["probability_of_loss"],
            }
        )

    return results


def make_walk_forward_splits(config: WalkForwardConfig) -> List[Dict[str, pd.Timestamp]]:
    """Split the full date span into sequential train, validation, and test folds."""
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    step_days = config.step_days or config.test_days
    cursor = start
    folds: List[Dict[str, pd.Timestamp]] = []

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=config.train_days - 1)
        val_start = train_end + pd.Timedelta(days=config.purge_gap_days + 1)
        val_end = val_start + pd.Timedelta(days=config.val_days - 1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=config.test_days - 1)

        if test_end > end:
            break

        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = cursor + pd.Timedelta(days=step_days)

    if not folds:
        raise ValueError("No valid walk-forward folds fit inside the requested date range")
    return folds
