"""Risk projection utilities for enforcing a maximum portfolio volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project an unconstrained vector onto the probability simplex."""
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D vector")
    sorted_weights = np.sort(weights)[::-1]
    cumulative = np.cumsum(sorted_weights)
    rho = np.nonzero(sorted_weights + (1.0 - cumulative) / (np.arange(len(weights)) + 1) > 0)[0][-1]
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(weights - theta, 0.0)
    return projected / projected.sum()


@dataclass
class RiskConstraint:
    """Projects target weights onto the simplex under a volatility cap."""

    sigma_max: float
    max_iterations: int = 500
    learning_rate: float = 0.05
    tolerance: float = 1e-6

    def project(self, weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Return the closest feasible weight vector found by projected descent."""
        covariance = np.asarray(covariance, dtype=np.float64)
        target = project_to_simplex(np.asarray(weights, dtype=np.float64))
        if self.portfolio_volatility(target, covariance) <= self.sigma_max:
            return target.astype(np.float32)

        candidate = target.copy()
        for _ in range(self.max_iterations):
            variance = float(candidate @ covariance @ candidate)
            volatility = np.sqrt(max(variance, 0.0))
            if volatility <= self.sigma_max + self.tolerance:
                break

            grad_distance = 2.0 * (candidate - target)
            grad_risk = np.zeros_like(candidate)
            if volatility > 0:
                grad_risk = (covariance @ candidate) / volatility
            candidate = candidate - self.learning_rate * (grad_distance + grad_risk)
            candidate = project_to_simplex(candidate)

        if self.portfolio_volatility(candidate, covariance) > self.sigma_max + self.tolerance:
            min_var = self._minimum_variance_portfolio(covariance, len(candidate))
            low, high = 0.0, 1.0
            for _ in range(64):
                mid = 0.5 * (low + high)
                blended = project_to_simplex(mid * candidate + (1.0 - mid) * min_var)
                if self.portfolio_volatility(blended, covariance) <= self.sigma_max:
                    low = mid
                else:
                    high = mid
            candidate = project_to_simplex(low * candidate + (1.0 - low) * min_var)

        return candidate.astype(np.float32)

    def portfolio_volatility(self, weights: np.ndarray, covariance: np.ndarray) -> float:
        """Compute the portfolio volatility implied by the covariance matrix."""
        variance = float(weights @ covariance @ weights)
        return float(np.sqrt(max(variance, 0.0)))

    def _minimum_variance_portfolio(self, covariance: np.ndarray, n_assets: int) -> np.ndarray:
        weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        for _ in range(self.max_iterations):
            grad = 2.0 * (covariance @ weights)
            updated = project_to_simplex(weights - self.learning_rate * grad)
            if np.linalg.norm(updated - weights, ord=2) < self.tolerance:
                weights = updated
                break
            weights = updated
        return weights

