"""Neural network components per Zhang et al. (2019) Exhibit 1."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer LSTM: 64 -> 32 units. Input (batch, seq_len, 10) -> (batch, 32)."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.act1 = nn.LeakyReLU()
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)
        out = self.act1(out)
        out, _ = self.lstm2(out)
        return out[:, -1, :]  # last hidden state: (batch, 32)


class DQNNetwork(nn.Module):
    """Dueling DQN: LSTM encoder -> value + advantage streams -> Q-values."""

    def __init__(self, n_features: int = 10, n_actions: int = 3):
        super().__init__()
        self.encoder = LSTMEncoder(n_features)
        self.value_head = nn.Linear(32, 1)
        self.advantage_head = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        v = self.value_head(h)                    # (batch, 1)
        a = self.advantage_head(h)                # (batch, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))  # (batch, n_actions)


class PGNetwork(nn.Module):
    """Policy Gradient (REINFORCE): LSTM encoder -> softmax action probs."""

    def __init__(self, n_features: int = 10, n_actions: int = 3):
        super().__init__()
        self.encoder = LSTMEncoder(n_features)
        self.head = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return torch.softmax(self.head(h), dim=-1)  # (batch, n_actions)


class A2CNetwork(nn.Module):
    """A2C: LSTM encoder -> actor (continuous mean via tanh) + critic (value)."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.encoder = LSTMEncoder(n_features)
        self.actor = nn.Linear(32, 1)
        self.critic = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean = torch.tanh(self.actor(h))   # (batch, 1), in [-1, 1]
        value = self.critic(h)              # (batch, 1)
        return mean.squeeze(-1), value.squeeze(-1)
