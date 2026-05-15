"""Standalone trainer for LSTM autoencoders with early stopping and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from portfolio_rl.encoding.asset_encoder.lstm_autoencoder import LSTMAutoencoder


@dataclass
class AutoencoderTrainingConfig:
    """Training hyperparameters for sequence autoencoder pretraining."""

    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-5
    device: str = "cpu"


class AutoencoderTrainer:
    """Encapsulates training, validation, and best-model persistence."""

    def __init__(
        self,
        model: LSTMAutoencoder,
        config: AutoencoderTrainingConfig,
        checkpoint_path: str | Path,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(self, train_sequences: np.ndarray, val_sequences: np.ndarray) -> Dict[str, List[float]]:
        """Train the model and return loss history for train and validation splits."""
        train_loader = self._make_loader(train_sequences, shuffle=True)
        val_loader = self._make_loader(val_sequences, shuffle=False)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_epoch = -1

        for epoch in range(self.config.max_epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                self._save_checkpoint(val_loss=val_loss, epoch=epoch)

            if epoch - best_epoch >= self.config.patience:
                break

        best_payload = torch.load(self.checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(best_payload["state_dict"])
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        total_examples = 0

        for (batch,) in loader:
            batch = batch.to(self.config.device)
            reconstruction = self.model(batch)
            loss = self.loss_fn(reconstruction, batch)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def _make_loader(self, sequences: np.ndarray, shuffle: bool) -> DataLoader:
        if sequences.ndim != 3:
            raise ValueError("Expected sequences with shape (N, T, F)")
        tensor = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, drop_last=False)

    def _save_checkpoint(self, val_loss: float, epoch: int) -> None:
        payload = self.model.checkpoint_payload()
        payload["val_loss"] = val_loss
        payload["epoch"] = epoch
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.checkpoint_path)

