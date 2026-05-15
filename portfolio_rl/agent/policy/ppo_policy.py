"""Proximal Policy Optimization implementation for portfolio allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from portfolio_rl.agent.policy.asset_scorer import AssetScorer
from portfolio_rl.agent.policy.base_policy import BasePolicy
from portfolio_rl.agent.policy.dirichlet_head import DirichletHead


@dataclass
class PPOConfig:
    """Configuration for PPO portfolio training."""

    state_dim: int
    num_assets: int
    latent_dim: int
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    device: str = "cpu"


class PPOActor(nn.Module):
    """Dirichlet actor built from shared asset-scoring layers."""

    def __init__(self, num_assets: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.latent_dim = latent_dim
        self.asset_scorer = AssetScorer(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.dirichlet_head = DirichletHead()

    def forward(self, asset_latents: torch.Tensor):
        scores = self.asset_scorer(asset_latents)
        return self.dirichlet_head(scores)


class PPOValueNetwork(nn.Module):
    """State-value baseline network for PPO."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOPolicy(BasePolicy):
    """PPO with Dirichlet-distributed portfolio allocations."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = PPOActor(config.num_assets, config.latent_dim, config.actor_hidden_dim).to(self.device)
        self.value_net = PPOValueNetwork(config.state_dim, config.critic_hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value_net.parameters()),
            lr=config.learning_rate,
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        action, _, _ = self.act(state, deterministic=False)
        return action

    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        asset_latents = self._extract_asset_latents(state_tensor)
        with torch.no_grad():
            dist = self.actor(asset_latents)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(state_tensor)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, batch: dict) -> float:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        num_samples = states.size(0)
        indices = np.arange(num_samples)

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                mb_states = states[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_returns = returns[batch_idx]
                mb_advantages = advantages[batch_idx]

                dist = self.actor(self._extract_asset_latents(mb_states))
                new_log_probs = dist.log_prob(mb_actions).unsqueeze(-1)
                entropy = dist.entropy().mean()
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                unclipped = ratios * mb_advantages
                clipped = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                clipped = clipped * mb_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()

                values = self.value_net(mb_states)
                value_loss = torch.mean((mb_returns - values) ** 2)
                loss = actor_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                total_loss += float(loss.item())

        batches = max((num_samples + self.config.minibatch_size - 1) // self.config.minibatch_size, 1)
        return total_loss / (self.config.update_epochs * batches)

    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: float,
    ) -> Dict[str, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            next_val = next_value if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.config.gamma * next_val * next_non_terminal - values[step]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        return {"advantages": advantages, "returns": returns}

    def _extract_asset_latents(self, state_tensor: torch.Tensor) -> torch.Tensor:
        latent_span = self.config.num_assets * self.config.latent_dim
        latents = state_tensor[:, :latent_span]
        return latents.view(-1, self.config.num_assets, self.config.latent_dim)

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "value_net": self.value_net.state_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.value_net.load_state_dict(payload["value_net"])
