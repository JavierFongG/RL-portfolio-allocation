"""Abstract policy interface for portfolio RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """Defines the minimal API shared by portfolio policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select a portfolio weight vector for the given state."""

    @abstractmethod
    def update(self, batch: dict) -> float:
        """Update the policy from a batch of experience and return a scalar loss."""

