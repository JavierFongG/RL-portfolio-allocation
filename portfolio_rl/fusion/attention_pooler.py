"""Attention pooling over variable-length sets of asset latent vectors."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooler(nn.Module):
    """Applies self-attention followed by learned pooling to latent sets."""

    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(latent_dim, 1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3:
            raise ValueError("latents must have shape (M, D) or (B, M, D)")

        key_padding_mask = None
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            key_padding_mask = ~mask.bool()

        attended, _ = self.attention(latents, latents, latents, key_padding_mask=key_padding_mask)
        attended = self.norm(attended + latents)

        scores = self.gate(attended).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(attended * weights, dim=1)
        return pooled.squeeze(0)

