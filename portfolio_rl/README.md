# portfolio_rl

`portfolio_rl` is the top-level package namespace for a modular reinforcement learning portfolio allocation system. Its role is to expose a clean separation between data engineering, representation learning, policy learning, environment simulation, and experiment orchestration.

## Design philosophy

The package is organized around the idea that portfolio RL is not one model but a pipeline of dependent modeling decisions:

1. How market data are sampled and cleaned.
2. How non-stationary series are normalized.
3. How temporal patterns are encoded into compact latent features.
4. How cross-sectional information across assets is aggregated.
5. How the action space is parameterized under portfolio constraints.
6. How the objective balances return, risk, turnover, and diversification.
7. How evaluation avoids look-ahead bias.

The directory split mirrors those questions directly.

## Subpackages

- `data/`: market data access and preprocessing.
- `encoding/`: sequence autoencoder used to learn per-asset latent states.
- `fusion/`: transformation from a set of asset latents into one environment observation.
- `agent/`: RL policies and reward functions.
- `environment/`: trading simulator and feasibility constraints.
- `training/`: experiment entrypoints.

## Why modularity matters here

In finance, the largest modeling errors often come from process mistakes rather than model architecture. A monolithic notebook can accidentally mix train and test statistics, refit scalers inside validation, or hide how turnover is penalized. A package architecture forces those assumptions into explicit modules, which makes the pipeline easier to audit and extend.

## Suggested reading order

1. `data/README.md`
2. `encoding/README.md`
3. `fusion/README.md`
4. `agent/README.md`
5. `environment/README.md`
6. `training/README.md`

## References

- Schulman et al. (2017), PPO: https://arxiv.org/abs/1707.06347
- Haarnoja et al. (2018), SAC: https://proceedings.mlr.press/v80/haarnoja18b.html
- Lopez de Prado (2018), financial ML validation: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
