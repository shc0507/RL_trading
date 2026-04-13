"""Q-network variants adapted from the homework DQN model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .torch_utils import build_mlp


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.q_head = build_mlp(
            input_size=state_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, state):
        return self.q_head(state)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        trunk_sizes = tuple(hidden_sizes)
        if trunk_sizes:
            trunk_output = int(trunk_sizes[-1])
            self.trunk = build_mlp(
                input_size=state_size,
                output_size=trunk_output,
                hidden_sizes=trunk_sizes[:-1],
                activation=activation,
            )
        else:
            trunk_output = state_size
            self.trunk = nn.Identity()
        self.value_head = build_mlp(
            input_size=trunk_output,
            output_size=1,
            hidden_sizes=(),
            activation=activation,
        )
        self.advantage_head = build_mlp(
            input_size=trunk_output,
            output_size=action_size,
            hidden_sizes=(),
            activation=activation,
        )

    def forward(self, state):
        latent = self.trunk(state)
        value = self.value_head(latent)
        advantage = self.advantage_head(latent)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class LSTMQNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        observation_window: int,
        feature_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
        recurrent_hidden_size: int = 128,
        recurrent_layers: int = 2,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.observation_window = int(observation_window)
        self.feature_size = int(feature_size)
        self.sequence_size = self.observation_window * self.feature_size
        expected_state_size = self.sequence_size + 1
        if int(state_size) != expected_state_size:
            raise ValueError(
                f"LSTMQNetwork expected state_size={expected_state_size}, got {state_size}"
            )
        self.dueling = bool(dueling)
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=int(recurrent_hidden_size),
            num_layers=max(1, int(recurrent_layers)),
            batch_first=True,
        )
        head_input_size = int(recurrent_hidden_size) + 1
        if self.dueling:
            self.value_head = build_mlp(
                input_size=head_input_size,
                output_size=1,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )
            self.advantage_head = build_mlp(
                input_size=head_input_size,
                output_size=action_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )
        else:
            self.q_head = build_mlp(
                input_size=head_input_size,
                output_size=action_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("LSTMQNetwork expects a 2D batch of flattened states")
        sequence = state[:, : self.sequence_size].reshape(
            state.shape[0],
            self.observation_window,
            self.feature_size,
        )
        position = state[:, self.sequence_size : self.sequence_size + 1]
        output, _ = self.lstm(sequence)
        last_hidden = output[:, -1, :]
        return torch.cat([last_hidden, position], dim=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent = self._encode(state)
        if not self.dueling:
            return self.q_head(latent)
        value = self.value_head(latent)
        advantage = self.advantage_head(latent)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(
    state_size: int,
    action_size: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: str = "relu",
    network_type: str = "mlp",
    dueling: bool = False,
    observation_window: int | None = None,
    feature_size: int | None = None,
    recurrent_hidden_size: int = 128,
    recurrent_layers: int = 2,
) -> nn.Module:
    normalized = network_type.lower()
    if normalized == "mlp":
        if dueling:
            return DuelingQNetwork(
                state_size=state_size,
                action_size=action_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )
        return QNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    if normalized == "lstm":
        if observation_window is None or feature_size is None:
            raise ValueError("observation_window and feature_size are required for LSTM DQN")
        return LSTMQNetwork(
            state_size=state_size,
            action_size=action_size,
            observation_window=observation_window,
            feature_size=feature_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            recurrent_hidden_size=recurrent_hidden_size,
            recurrent_layers=recurrent_layers,
            dueling=dueling,
        )
    raise ValueError(f"unsupported DQN network_type {network_type}")
