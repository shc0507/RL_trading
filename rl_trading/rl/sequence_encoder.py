"""Shared recurrent state encoders for Zhang-style models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def normalize_recurrent_layer_sizes(
    recurrent_hidden_size: int = 128,
    recurrent_layers: int = 2,
    recurrent_layer_sizes: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if recurrent_layer_sizes is not None:
        sizes = tuple(int(size) for size in recurrent_layer_sizes)
        if not sizes:
            raise ValueError("recurrent_layer_sizes must not be empty")
        if any(size <= 0 for size in sizes):
            raise ValueError("recurrent_layer_sizes must be positive")
        return sizes
    if recurrent_layers <= 0 or recurrent_hidden_size <= 0:
        raise ValueError("recurrent_hidden_size and recurrent_layers must be positive")
    return tuple(int(recurrent_hidden_size) for _ in range(int(recurrent_layers)))


class StackedLSTMStateEncoder(nn.Module):
    """Encodes flattened trading states into a recurrent latent vector plus current position."""

    def __init__(
        self,
        state_size: int,
        observation_window: int,
        feature_size: int,
        recurrent_layer_sizes: Sequence[int] = (64, 32),
        recurrent_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.observation_window = int(observation_window)
        self.feature_size = int(feature_size)
        self.recurrent_layer_sizes = tuple(int(size) for size in recurrent_layer_sizes)
        self.recurrent_dropout = float(recurrent_dropout)
        self.output_dropout = float(output_dropout)
        self.sequence_size = self.observation_window * self.feature_size
        expected_state_size = self.sequence_size + 1
        if int(state_size) != expected_state_size:
            raise ValueError(
                f"StackedLSTMStateEncoder expected state_size={expected_state_size}, got {state_size}"
            )
        input_size = self.feature_size
        layers: list[nn.Module] = []
        for hidden_size in self.recurrent_layer_sizes:
            layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True))
            input_size = hidden_size
        self.lstm_layers = nn.ModuleList(layers)
        self.dropout_layers = nn.ModuleList(
            [
                nn.Dropout(self.recurrent_dropout) if self.recurrent_dropout > 0.0 else nn.Identity()
                for _ in self.recurrent_layer_sizes
            ]
        )
        self.output_dropout_layer = nn.Dropout(self.output_dropout) if self.output_dropout > 0.0 else nn.Identity()
        self.output_size = input_size + 1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("StackedLSTMStateEncoder expects a 2D batch of flattened states")
        sequence = state[:, : self.sequence_size].reshape(
            state.shape[0],
            self.observation_window,
            self.feature_size,
        )
        position = state[:, self.sequence_size : self.sequence_size + 1]
        output = sequence
        for layer, dropout in zip(self.lstm_layers, self.dropout_layers, strict=True):
            output, _ = layer(output)
            output = dropout(output)
        last_hidden = output[:, -1, :]
        last_hidden = self.output_dropout_layer(last_hidden)
        return torch.cat([last_hidden, position], dim=1)
