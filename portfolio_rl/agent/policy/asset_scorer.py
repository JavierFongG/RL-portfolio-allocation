"""Shared-weight per-asset scoring network for latent asset embeddings."""

from __future__ import annotations

import torch
from torch import nn


class AssetScorer(nn.Module):
    """Applies the same MLP to each asset latent to produce scalar scores."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []
        input_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3:
            raise ValueError("latents must have shape (M, D) or (B, M, D)")
        batch_size, num_assets, latent_dim = latents.shape
        scores = self.network(latents.reshape(batch_size * num_assets, latent_dim))
        return scores.view(batch_size, num_assets)

