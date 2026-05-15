# Asset Encoder

This folder implements the asset-level sequence autoencoder used to convert a feature window for one asset into a latent factor vector.

## Files

### `lstm_autoencoder.py`

Defines:

- `LSTMAutoencoderConfig`
- `LSTMEncoder`
- `LSTMDecoder`
- `LSTMAutoencoder`

The encoder processes a sequence and produces a latent vector `z`. The decoder maps `z` back into a reconstructed sequence. The encoder is kept directly accessible through `encode()` and through the standalone `LSTMEncoder` class because downstream RL only needs inference-time latent extraction.

### `autoencoder_trainer.py`

Implements supervised training of the autoencoder with mean squared reconstruction loss, mini-batch optimization, validation tracking, early stopping, and best-checkpoint saving.

### `encoder_registry.py`

Loads a saved checkpoint, reconstructs the architecture from stored config metadata, freezes all parameters, and returns a ready-to-use encoder module.

## Why this encoder architecture

### Stacked LSTM

A stacked recurrent encoder is a practical middle ground:

- more expressive than a single recurrent layer
- simpler and lighter than a large attention model
- naturally compatible with fixed windows

### Linear projection to latent space

The final linear projection separates the recurrent hidden size from the latent dimension. That makes it easier to control the size of the representation exposed to the policy.

### Reconstruction objective

The reconstruction loss is not a direct portfolio objective, but it provides a dense training signal. In a low-sample financial setting, that is valuable because pure RL feedback is sparse and noisy.

### Early stopping

Autoencoders can overfit easily, especially when the latent dimension is not very small. Early stopping on a held-out validation split is a practical regularization device.

### Frozen inference-time encoder

Freezing the encoder during RL avoids two common failure modes:

- catastrophic drift in the latent representation while the policy is learning
- leakage from reusing test-time data in representation fitting

## References

- Hochreiter and Schmidhuber (1997), LSTM: https://doi.org/10.1162/neco.1997.9.8.1735
- Goodfellow, Bengio, and Courville (2016), representation learning overview: https://www.deeplearningbook.org/
- PyTorch saving and loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
