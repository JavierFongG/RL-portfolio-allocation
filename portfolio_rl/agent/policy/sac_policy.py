"""Soft Actor-Critic implementation for simplex-constrained portfolio actions."""

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
class SACConfig:
    """Configuration for the portfolio SAC policy."""

    state_dim: int
    num_assets: int
    latent_dim: int
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    alpha_lr: float = 3e-4
    replay_capacity: int = 100_000
    batch_size: int = 128
    target_entropy: float | None = None
    device: str = "cpu"


class ReplayBuffer:
    """Simple numpy replay buffer for off-policy training."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.position] = state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.done[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.state[indices],
            "actions": self.action[indices],
            "rewards": self.reward[indices],
            "next_states": self.next_state[indices],
            "dones": self.done[indices],
        }


class ActorNetwork(nn.Module):
    """Actor that scores assets independently and emits a Dirichlet distribution."""

    def __init__(self, num_assets: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.latent_dim = latent_dim
        self.asset_scorer = AssetScorer(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.dirichlet_head = DirichletHead()

    def forward(self, asset_latents: torch.Tensor):
        scores = self.asset_scorer(asset_latents)
        return self.dirichlet_head(scores), scores


class CriticNetwork(nn.Module):
    """Q-network over state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))


class ValueNetwork(nn.Module):
    """State-value network used by the SAC update."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class SACPolicy(BasePolicy):
    """Soft Actor-Critic with Dirichlet portfolio actions."""

    def __init__(self, config: SACConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = ActorNetwork(config.num_assets, config.latent_dim, config.actor_hidden_dim).to(self.device)
        self.critic_1 = CriticNetwork(config.state_dim, config.num_assets, config.critic_hidden_dim).to(self.device)
        self.critic_2 = CriticNetwork(config.state_dim, config.num_assets, config.critic_hidden_dim).to(self.device)
        self.value_net = ValueNetwork(config.state_dim, config.critic_hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(config.state_dim, config.critic_hidden_dim).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)

        initial_alpha = np.log(0.1)
        self.log_alpha = torch.tensor(initial_alpha, dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(config.num_assets)
        self.replay_buffer = ReplayBuffer(config.state_dim, config.num_assets, config.replay_capacity)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        action, _, _ = self.act(state, deterministic=False)
        return action

    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        asset_latents = self._extract_asset_latents(state_tensor)
        with torch.no_grad():
            distribution, scores = self.actor(asset_latents)
            action = distribution.mean if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
            value = self.value_net(state_tensor)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, batch: dict) -> float:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_values = self.target_value_net(next_states)
            q_targets = rewards + self.config.gamma * (1.0 - dones) * target_values

        q1_loss = torch.mean((self.critic_1(states, actions) - q_targets) ** 2)
        q2_loss = torch.mean((self.critic_2(states, actions) - q_targets) ** 2)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.critic_2_optimizer.step()

        asset_latents = self._extract_asset_latents(states)
        distribution, scores = self.actor(asset_latents)
        sampled_actions = distribution.rsample()
        log_probs = distribution.log_prob(sampled_actions).unsqueeze(-1)
        q_min = torch.minimum(
            self.critic_1(states, sampled_actions),
            self.critic_2(states, sampled_actions),
        )
        value_target = (q_min - self.alpha.detach() * log_probs).detach()
        value_loss = torch.mean((self.value_net(states) - value_target) ** 2)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = torch.mean(self.alpha.detach() * log_probs - q_min)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.value_net, self.target_value_net)
        total_loss = q1_loss + q2_loss + value_loss + actor_loss + alpha_loss
        return float(total_loss.item())

    def _extract_asset_latents(self, state_tensor: torch.Tensor) -> torch.Tensor:
        latent_span = self.config.num_assets * self.config.latent_dim
        latents = state_tensor[:, :latent_span]
        return latents.view(-1, self.config.num_assets, self.config.latent_dim)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data
            )

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "value_net": self.value_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic_1.load_state_dict(payload["critic_1"])
        self.critic_2.load_state_dict(payload["critic_2"])
        self.value_net.load_state_dict(payload["value_net"])
        self.target_value_net.load_state_dict(payload["target_value_net"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))

