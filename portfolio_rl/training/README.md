# Training

This folder contains the experiment entrypoints that connect the rest of the package into a reproducible workflow.

## Files

### `pretrain_encoder.py`

Builds aligned and normalized sequence datasets for the full asset universe, trains the LSTM autoencoder on the training split, validates it on a separate split, and saves the best checkpoint.

### `train_agent.py`

Loads the frozen encoder, creates the state builder, environment, reward, and selected RL policy, runs the training loop, performs deterministic held-out evaluation, runs Monte Carlo robustness evaluation, logs episode-level metrics, and saves the trained policy.

### `walk_forward.py`

Creates sequential train, validation, and test folds with a purge gap, then runs encoder pretraining and RL training on each fold. The output is an out-of-sample evaluation trace across time enriched with Monte Carlo robustness summaries for each fold.

### `monte_carlo.py`

Runs repeated stochastic rollouts of a trained policy on the held-out environment and summarizes the distribution of outcomes. It reports metrics such as mean and interval estimates for Sharpe and return, plus drawdown, turnover, final wealth, and probability of loss.

## Why the training workflow is split

### Separate encoder pretraining

This prevents the representation model from absorbing future information through repeated refits during evaluation. It also makes experiments easier to compare because one can hold the encoder fixed and swap only the RL algorithm or reward.

### Walk-forward evaluation

Random train-test splits are usually inappropriate for financial time series because:

- time dependence matters
- overlapping windows create leakage risk
- regime shifts make IID assumptions unrealistic

A walk-forward design better matches how a live system would actually be deployed.

### Monte Carlo robustness evaluation

One deterministic evaluation path is often not enough when the learned actor is stochastic. In this package, PPO and SAC use Dirichlet-based portfolio policies, so their sampled allocations can vary even under the same market history. The Monte Carlo module evaluates this by running many stochastic rollouts on the held-out environment and collecting a distribution of outcomes rather than a single point estimate.

This helps answer questions like:

- Is the policy consistently good, or just good on its deterministic mean action?
- How wide is the distribution of Sharpe and total return under policy sampling?
- How often does the policy lose money out of sample?
- Does stochastic exploration lead to materially different turnover or drawdown behavior?

The current implementation is policy-side Monte Carlo rather than market-side Monte Carlo. That means it samples different actions from the learned stochastic policy on the same held-out market path. This is useful for assessing action robustness, though it is not yet a full scenario generator for alternative market histories.

### Purge gap

The purge gap reduces contamination between training and validation when adjacent observations share temporal context. This is especially relevant when windows overlap or when target behavior depends on recent history.

## Recommended experiment interpretation

When reading results, do not rely on one metric. At minimum inspect:

- out-of-sample Sharpe
- total return
- turnover
- concentration
- stability across folds
- Monte Carlo confidence ranges for Sharpe and return
- probability of loss across stochastic evaluation rollouts
- average drawdown and turnover dispersion under stochastic sampling

A policy that performs well only in one fold or only under high turnover is usually less convincing than one that is moderate but consistent.

## References

- Lopez de Prado (2018), purging and embargo concepts: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- PyTorch training utilities: https://pytorch.org/docs/stable/index.html
