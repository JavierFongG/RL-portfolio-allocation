"""Encoder pretraining pipeline that fits an LSTM autoencoder on train-only normalized sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from portfolio_rl.data.loaders.registry import get_loader
from portfolio_rl.data.preprocessing.alignment import align_frames
from portfolio_rl.data.preprocessing.normalizer import RollingZScoreNormalizer
from portfolio_rl.data.preprocessing.sequencer import create_sequences
from portfolio_rl.encoding.asset_encoder.autoencoder_trainer import (
    AutoencoderTrainer,
    AutoencoderTrainingConfig,
)
from portfolio_rl.encoding.asset_encoder.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig
from portfolio_rl.fusion.state_builder import AssetSpec


@dataclass
class PretrainEncoderConfig:
    """Configuration for standalone encoder pretraining."""

    asset_specs: Sequence[AssetSpec]
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    window_size: int
    hidden_dim: int
    latent_dim: int
    normalizer_window: int = 20
    feature_columns: Sequence[str] = ("open", "high", "low", "close", "volume")
    checkpoint_path: str = "artifacts/encoder.pt"
    trainer: AutoencoderTrainingConfig = field(default_factory=AutoencoderTrainingConfig)


def run_pretraining(config: PretrainEncoderConfig) -> Dict[str, object]:
    """Pretrain the autoencoder and return training metadata."""
    train_start = pd.Timestamp(config.train_start)
    train_end = pd.Timestamp(config.train_end)
    val_start = pd.Timestamp(config.val_start)
    val_end = pd.Timestamp(config.val_end)
    raw_frames = _load_price_frames(config.asset_specs, train_start, val_end)

    aligned_frames = align_frames(raw_frames.values(), max_forward_fill=3)
    aligned_by_ticker = {
        asset.ticker: frame for asset, frame in zip(config.asset_specs, aligned_frames)
    }

    train_sequences = []
    val_sequences = []
    normalizers: Dict[str, RollingZScoreNormalizer] = {}

    for asset in config.asset_specs:
        frame = aligned_by_ticker[asset.ticker]
        frame.columns = frame.columns.droplevel(1)  # drop the ticker level
        frame.loc[:, config.feature_columns]
        train_frame = frame.loc[(frame.index >= train_start) & (frame.index <= train_end)]
        val_frame = frame.loc[(frame.index >= val_start) & (frame.index <= val_end)]
        if len(train_frame) < config.window_size or len(val_frame) < config.window_size:
            raise ValueError(f"Not enough rows to build sequences for asset {asset.ticker!r}")

        normalizer = RollingZScoreNormalizer(window_size=config.normalizer_window).fit(train_frame)
        normalizers[asset.ticker] = normalizer
        normalized_train = normalizer.transform(train_frame)
        normalized_val = normalizer.transform(val_frame)
        train_sequences.append(create_sequences(normalized_train, config.window_size))
        val_sequences.append(create_sequences(normalized_val, config.window_size))

    train_array = np.concatenate(train_sequences, axis=0)
    val_array = np.concatenate(val_sequences, axis=0)

    model = LSTMAutoencoder(
        LSTMAutoencoderConfig(
            input_dim=len(config.feature_columns),
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )
    )
    trainer = AutoencoderTrainer(model=model, config=config.trainer, checkpoint_path=config.checkpoint_path)
    history = trainer.fit(train_sequences=train_array, val_sequences=val_array)

    return {
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "history": history,
        "num_train_sequences": int(len(train_array)),
        "num_val_sequences": int(len(val_array)),
        "normalizers": normalizers,
    }


def _load_price_frames(
    asset_specs: Sequence[AssetSpec],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    fetch_start = (start - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    fetch_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames: Dict[str, pd.DataFrame] = {}
    for asset in asset_specs:
        loader = get_loader(asset.asset_type)
        frames[asset.ticker] = loader.fetch(asset.ticker, fetch_start, fetch_end)
    return frames

