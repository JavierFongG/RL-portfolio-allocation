"""Dirichlet action head for simplex-constrained portfolio weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Dirichlet


class DirichletHead(nn.Module):
    """Maps unconstrained asset scores to a Dirichlet distribution."""

    def __init__(self, min_concentration: float = 1e-3) -> None:
        super().__init__()
        self.min_concentration = min_concentration
        self.softplus = nn.Softplus()

    def forward(self, scores: torch.Tensor) -> Dirichlet:
        concentrations = self.softplus(scores) + self.min_concentration
        return Dirichlet(concentrations)

    def sample(self, scores: torch.Tensor, reparameterize: bool = False) -> torch.Tensor:
        distribution = self.forward(scores)
        return distribution.rsample() if reparameterize else distribution.sample()

    def log_prob(self, scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(scores)
        return distribution.log_prob(weights)

