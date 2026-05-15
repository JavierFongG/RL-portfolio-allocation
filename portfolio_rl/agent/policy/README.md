# Policy

This folder implements action selection for the portfolio agent.

## Files

### `base_policy.py`

Defines the abstract interface shared by all policies:

- `select_action(state)`
- `update(batch)`

This keeps training code independent of the specific RL algorithm.

### `asset_scorer.py`

Applies the same MLP to each asset latent vector independently. This is a weight-sharing design: every asset is scored by the same function, which is appropriate when asset identity should not imply a completely different architecture.

### `dirichlet_head.py`

Transforms per-asset scores into a Dirichlet distribution over portfolio weights. This gives a stochastic policy whose support is exactly the long-only simplex.

### `ppo_policy.py`

Implements Proximal Policy Optimization with:

- clipped policy objective
- value function baseline
- generalized advantage estimation
- entropy regularization

PPO is a strong baseline because it is relatively robust and straightforward to tune.

### `sac_policy.py`

Implements Soft Actor-Critic with:

- actor network
- twin critics
- value and target value networks
- entropy temperature tuning
- replay buffer

SAC is useful when off-policy reuse of experience matters.

## Theory behind the main decisions

### Shared-weight asset scoring

The `AssetScorer` reflects a permutation-aware intuition: what matters is the latent structure of each asset, not a separate hand-crafted network for ticker number one versus ticker number five. Shared weights reduce parameter count and improve scalability to different asset universes.

### Dirichlet policy head

The Dirichlet distribution is a natural choice for a long-only fully invested portfolio because:

- every sample is non-negative
- the weights sum to one
- concentration can be tuned through the concentration parameters

This is cleaner than sampling unconstrained Gaussians and projecting afterward, which can create a mismatch between the distribution being optimized and the action actually executed.

### PPO versus SAC

PPO is attractive when one wants a stable baseline with direct trajectory optimization and GAE. SAC is attractive when sample reuse matters and entropy-regularized exploration is desirable. Including both is useful for research comparison because portfolio RL can be sensitive to turnover, delayed effects, and state noise.

## References

- Schulman et al. (2017), PPO: https://arxiv.org/abs/1707.06347
- Haarnoja et al. (2018), SAC: https://proceedings.mlr.press/v80/haarnoja18b.html
- Sadeghi and Banihashemi (2022), Dirichlet policy for portfolio RL: https://www.mdpi.com/2075-1680/11/12/664
- Costa and Aaltonen (2020), reinforced factor portfolios with Dirichlet policies: https://arxiv.org/abs/2011.05381
