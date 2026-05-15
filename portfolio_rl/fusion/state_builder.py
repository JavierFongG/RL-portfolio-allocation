"""State builder that fetches, normalizes, encodes, and pools asset histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from portfolio_rl.data.loaders.registry import get_loader
from portfolio_rl.data.preprocessing.normalizer import RollingZScoreNormalizer
from portfolio_rl.encoding.asset_encoder.lstm_autoencoder import LSTMEncoder
from portfolio_rl.fusion.attention_pooler import AttentionPooler


@dataclass
class AssetSpec:
    """Describes how to fetch and identify a tradable asset."""

    ticker: str
    asset_type: str


@dataclass
class StateBuilder:
    """Builds state vectors from asset histories and a frozen encoder."""

    encoder: LSTMEncoder
    attention_pooler: AttentionPooler
    normalizers: Mapping[str, RollingZScoreNormalizer]
    window_size: int
    feature_columns: Sequence[str]
    device: str = "cpu"
    data_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def build_state(
        self,
        as_of_date: str | pd.Timestamp,
        asset_subset: Iterable[AssetSpec],
        risk_scalar: float,
    ) -> np.ndarray:
        """Return a flattened state vector containing asset latents, pooled context, and risk."""
        timestamp = pd.Timestamp(as_of_date)
        asset_latents: List[np.ndarray] = []

        self.encoder.eval()
        self.attention_pooler.eval()
        with torch.no_grad():
            for asset in asset_subset:
                sequence = self._fetch_sequence(asset, timestamp)
                normalizer = self.normalizers[asset.ticker]
                normalized = normalizer.transform(sequence.loc[:, self.feature_columns])
                window = normalized.tail(self.window_size).to_numpy(dtype=np.float32)
                if len(window) != self.window_size:
                    raise ValueError(
                        f"Insufficient normalized history for {asset.ticker!r} at {timestamp.date()}"
                    )
                tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
                latent = self.encoder(tensor).squeeze(0).cpu().numpy()
                asset_latents.append(latent)

            latent_matrix = np.stack(asset_latents, axis=0)
            pooled = self.attention_pooler(
                torch.tensor(latent_matrix, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        return np.concatenate(
            [latent_matrix.reshape(-1), pooled.astype(np.float32), np.array([risk_scalar], dtype=np.float32)]
        ).astype(np.float32)

    def _fetch_sequence(self, asset: AssetSpec, as_of_date: pd.Timestamp) -> pd.DataFrame:
        if asset.ticker in self.data_cache:
            history = self.data_cache[asset.ticker]
        else:
            loader = get_loader(asset.asset_type)
            history = loader.fetch(
                ticker=asset.ticker,
                start=(as_of_date - pd.Timedelta(days=self.window_size * 5)).strftime("%Y-%m-%d"),
                end=(as_of_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            self.data_cache[asset.ticker] = history

        history = history.loc[history.index <= as_of_date]
        if len(history) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} rows for asset {asset.ticker!r}")
        return history.copy()
