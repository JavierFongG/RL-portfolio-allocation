"""Checkpoint loader that restores and freezes trained asset encoders."""

from __future__ import annotations

from pathlib import Path

import torch

from portfolio_rl.encoding.asset_encoder.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
    LSTMEncoder,
)


class EncoderRegistry:
    """Loads frozen encoders from saved autoencoder checkpoints."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._encoder: LSTMEncoder | None = None

    def get_encoder(self) -> LSTMEncoder:
        """Return a frozen encoder ready for inference."""
        if self._encoder is None:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            config = LSTMAutoencoderConfig(**payload["config"])
            autoencoder = LSTMAutoencoder(config)
            autoencoder.load_state_dict(payload["state_dict"])
            encoder = autoencoder.encoder.to(self.device)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            self._encoder = encoder
        return self._encoder

