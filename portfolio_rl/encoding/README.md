# Encoding

This folder contains representation-learning components that transform per-asset time-series windows into dense latent vectors.

## Why representation learning is separated from RL

Training the sequence encoder independently has two benefits:

1. It reduces the dimensional burden on the RL policy.
2. It prevents the policy from having to learn temporal compression and allocation behavior simultaneously from a sparse reward signal.

This is a common strategy when the observation space is high-dimensional and the control problem is harder than the reconstruction problem.

## Subfolder

### `asset_encoder/`

Contains the LSTM autoencoder, the training loop, and checkpoint loading utilities.

## Theoretical motivation

An autoencoder learns a compressed representation that preserves information needed to reconstruct the original signal. In this project, reconstruction is used as a self-supervised objective: price sequences do not need labels, so the encoder can be trained before the RL stage.

An LSTM is used instead of a plain feed-forward encoder because:

- order matters inside each lookback window
- financial features may contain lagged dependencies
- gating helps preserve medium-range sequential information better than a shallow recurrent baseline

This is not the only possible choice. A Transformer or temporal convolution could also work, but the LSTM is a strong baseline with lower implementation overhead.

## References

- Hochreiter and Schmidhuber (1997), LSTM: https://doi.org/10.1162/neco.1997.9.8.1735
- PyTorch recurrent module documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
