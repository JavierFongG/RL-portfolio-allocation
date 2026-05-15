# Training

This folder contains the experiment entrypoints that connect the rest of the package into a reproducible workflow.

## Files

### `pretrain_encoder.py`

Builds aligned and normalized sequence datasets for the full asset universe, trains the LSTM autoencoder on the training split, validates it on a separate split, and saves the best checkpoint.

### `train_agent.py`

Loads the frozen encoder, creates the state builder, environment, reward, and selected RL policy, runs the training loop, logs episode-level metrics, and saves the trained policy.

### `walk_forward.py`

Creates sequential train, validation, and test folds with a purge gap, then runs encoder pretraining and RL training on each fold. The output is an out-of-sample evaluation trace across time.

## Why the training workflow is split

### Separate encoder pretraining

This prevents the representation model from absorbing future information through repeated refits during evaluation. It also makes experiments easier to compare because one can hold the encoder fixed and swap only the RL algorithm or reward.

### Walk-forward evaluation

Random train-test splits are usually inappropriate for financial time series because:

- time dependence matters
- overlapping windows create leakage risk
- regime shifts make IID assumptions unrealistic

A walk-forward design better matches how a live system would actually be deployed.

### Purge gap

The purge gap reduces contamination between training and validation when adjacent observations share temporal context. This is especially relevant when windows overlap or when target behavior depends on recent history.

## Recommended experiment interpretation

When reading results, do not rely on one metric. At minimum inspect:

- out-of-sample Sharpe
- total return
- turnover
- concentration
- stability across folds

A policy that performs well only in one fold or only under high turnover is usually less convincing than one that is moderate but consistent.

## References

- Lopez de Prado (2018), purging and embargo concepts: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- PyTorch training utilities: https://pytorch.org/docs/stable/index.html
