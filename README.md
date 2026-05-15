# RL Portfolio Allocation

This repository contains `portfolio_rl`, a research-oriented Python package for reinforcement learning based portfolio allocation. The package is intentionally modular: market data ingestion, preprocessing, sequence encoding, state fusion, policy learning, reward design, environment dynamics, and experiment orchestration live in separate subpackages so that each layer can be replaced or audited independently.

## Why the package is structured this way

Portfolio RL systems are fragile when data handling, representation learning, and policy learning are mixed into one training loop. This package separates those responsibilities for three reasons:

1. Leakage control. Training-only transformations such as normalization and encoder fitting must never see validation or test data.
2. Research clarity. It should be possible to change the reward, the encoder, or the policy algorithm without rewriting the rest of the stack.
3. Financial realism. State construction, turnover penalties, and risk constraints need explicit implementation rather than being hidden inside a generic RL pipeline.

## Package map

- `portfolio_rl/data/`: loaders and preprocessing for OHLCV data.
- `portfolio_rl/encoding/`: asset-level sequence representation learning.
- `portfolio_rl/fusion/`: cross-asset aggregation into a policy state.
- `portfolio_rl/agent/`: policy algorithms and reward definitions.
- `portfolio_rl/environment/`: Gymnasium environment and portfolio constraints.
- `portfolio_rl/training/`: encoder pretraining, RL training, walk-forward evaluation, and Monte Carlo robustness analysis.

Each folder contains its own `README.md` with more detail.

## Core modeling decisions

### 1. Price histories are transformed before policy learning

Raw OHLCV features have very different scales across assets and through time. A rolling z-score normalizer is used because financial time series are non-stationary: the relevant scale of volatility and volume is local rather than global. Using a rolling statistic is a compromise between full online normalization and a static transform.

### 2. Asset histories are compressed with an LSTM autoencoder

The encoder maps a multivariate window of market features into a latent vector per asset. The reason is dimensionality control: without compression, the policy must directly process long windows for many assets, which increases sample complexity and encourages overfitting. An LSTM was chosen because the input is sequential and ordering matters.

### 3. Cross-asset state is built in two stages

The state builder keeps both:

- per-asset latent vectors, flattened into the observation so the policy can score assets individually
- a pooled attention summary, which acts as a market-level context vector

This design avoids forcing a single pooled summary to carry all cross-sectional information, while still giving the policy a global view of the current market configuration.

### 4. Actions are distributions over the simplex

Portfolio weights must be non-negative and sum to one in the current implementation. A Dirichlet head is a natural way to represent stochastic policies on the simplex because its support already matches the feasible action space. That avoids post hoc normalization tricks inside the actor.

### 5. Reward is portfolio-return-first but risk-aware

The main reward uses log portfolio return and subtracts normalized penalties for:

- realized volatility
- turnover
- concentration via the Herfindahl-Hirschman Index

The normalization uses an exponential moving average of absolute term magnitudes so the penalties remain comparable even if one term naturally has a smaller raw numerical scale than another.

### 6. Evaluation is walk-forward and stress-tested with Monte Carlo rollouts

Financial data are time ordered, autocorrelated, and vulnerable to leakage through overlapping labels and transformations. The walk-forward pipeline keeps train, validation, and test periods sequential and includes a purge gap to reduce contamination between phases. After training, the held-out test environment is also evaluated with repeated stochastic policy rollouts so we can inspect the distribution of outcomes induced by the learned Dirichlet policy rather than relying only on one deterministic backtest path.

## Practical workflow

1. Load and align asset histories.
2. Fit train-only rolling normalizers.
3. Build fixed-length windows from training data.
4. Pretrain the LSTM autoencoder on train sequences and validate with early stopping.
5. Freeze the encoder.
6. Build environment states from normalized windows, encoded asset latents, attention pooling, and risk metadata.
7. Train an RL policy with PPO or SAC.
8. Evaluate out of sample with deterministic test episodes and walk-forward folds.
9. Run Monte Carlo rollouts on the held-out environment to estimate robustness ranges for Sharpe, return, drawdown, and turnover.

## What is intentionally simplified

This package is built as a clear baseline, not a full institutional backtesting engine. Some simplifications are deliberate:

- the current loader uses Yahoo Finance through `yfinance`
- action feasibility is modeled on the long-only simplex
- transaction costs are linear in turnover
- risk projection is approximate and uses projected descent rather than a dedicated convex solver
- reward penalties are based on realized signals rather than forecast risk models
- Monte Carlo robustness currently samples alternative policy actions on the same held-out market path rather than generating fully resampled market scenarios

Those choices keep the code dependency-light and readable while preserving the main research ideas.

## Folder-level reading guide

- Start with `portfolio_rl/data/README.md` to understand leakage control.
- Then read `portfolio_rl/encoding/README.md` and `portfolio_rl/fusion/README.md` for the representation pipeline.
- Read `portfolio_rl/agent/README.md` and `portfolio_rl/environment/README.md` together to understand the RL objective and dynamics.
- Finish with `portfolio_rl/training/README.md` for the experiment lifecycle.

## Bibliography and references

### Reinforcement learning

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv. https://arxiv.org/abs/1707.06347
2. Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. PMLR 80. https://proceedings.mlr.press/v80/haarnoja18b.html

### Sequence modeling and attention

3. Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762

### Portfolio and risk

5. Sharpe, W. F. (1994). *The Sharpe Ratio*. The Journal of Portfolio Management, 21(1), 49-58. https://doi.org/10.3905/jpm.1994.409501
6. Rockafellar, R. T., and Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21-41. https://doi.org/10.21314/JOR.2000.038
7. U.S. Department of Justice. *Herfindahl-Hirschman Index*. https://www.justice.gov/atr/herfindahl-hirschman-index

### Financial machine learning and validation

8. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

### Tooling

9. Gymnasium documentation. https://gymnasium.farama.org/
10. PyTorch documentation. https://pytorch.org/docs/stable/index.html
