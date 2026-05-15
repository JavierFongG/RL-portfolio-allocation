# Agent

This folder contains the decision-making layer: portfolio policies and reward definitions.

## Why policy and reward are grouped here

In reinforcement learning, the policy answers “what action should be taken?” and the reward answers “what behavior is desirable?” In portfolio management those two questions are tightly linked, because a different reward specification changes the kind of portfolio the policy learns to prefer.

## Subfolders

### `policy/`

Contains the action-distribution parameterization and the PPO and SAC implementations.

### `reward/`

Contains reward abstractions and concrete portfolio objectives.

## Design principles

### Simplex-aware actions

Portfolio weights are constrained objects. The agent should model them with a distribution defined on the simplex instead of producing unconstrained outputs and normalizing them afterward as an afterthought.

### Risk-aware reward

Raw return is not a sufficient objective in portfolio learning. A trading policy that maximizes one-step return without regard to volatility, turnover, or concentration often learns unstable and unrealistic behavior. The reward layer therefore includes multiple finance-motivated definitions.

### Multiple algorithms

The package includes PPO and SAC because they represent complementary regimes:

- PPO is on-policy and typically simpler to stabilize in fresh research setups
- SAC is off-policy and can be more sample-efficient when replay helps

## References

- Schulman et al. (2017), PPO: https://arxiv.org/abs/1707.06347
- Haarnoja et al. (2018), SAC: https://proceedings.mlr.press/v80/haarnoja18b.html
