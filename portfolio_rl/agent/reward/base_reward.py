"""Abstract reward interface for portfolio allocation objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseReward(ABC):
    """Defines the reward contract expected by the environment."""

    def reset(self) -> None:
        """Reset any internal state accumulated across environment steps."""

    @abstractmethod
    def compute(self, weights: np.ndarray, returns: np.ndarray, prev_weights: np.ndarray):
        """Compute reward outputs from the current portfolio transition."""
