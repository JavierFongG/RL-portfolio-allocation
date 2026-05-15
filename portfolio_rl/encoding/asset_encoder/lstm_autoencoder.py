"""PyTorch LSTM autoencoder for compressing asset feature sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class LSTMAutoencoderConfig:
    """Hyperparameters required to construct the autoencoder."""

    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_layers: int = 2
    dropout: float = 0.0


class LSTMEncoder(nn.Module):
    """Encodes a sequence into a latent vector using stacked LSTMs."""

    def __init__(self, config: LSTMAutoencoderConfig) -> None:
        super().__init__()
        dropout = config.dropout if config.num_layers > 1 else 0.0
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        top_hidden = hidden[-1]
        return self.projection(top_hidden)


class LSTMDecoder(nn.Module):
    """Reconstructs sequences from latent vectors through an LSTM decoder."""

    def __init__(self, config: LSTMAutoencoderConfig) -> None:
        super().__init__()
        dropout = config.dropout if config.num_layers > 1 else 0.0
        self.config = config
        self.latent_to_hidden = nn.Linear(config.latent_dim, config.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(config.hidden_dim, config.input_dim)

    def forward(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        decoder_seed = self.latent_to_hidden(z).unsqueeze(1)
        repeated_seed = decoder_seed.repeat(1, sequence_length, 1)
        decoded, _ = self.lstm(repeated_seed)
        return self.output_layer(decoded)


class LSTMAutoencoder(nn.Module):
    """End-to-end sequence autoencoder with a reusable encoder interface."""

    def __init__(self, config: LSTMAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z, sequence_length=x.size(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def checkpoint_payload(self) -> dict:
        """Return the metadata needed to reconstruct this model from disk."""
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

