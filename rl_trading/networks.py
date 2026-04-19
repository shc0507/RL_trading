"""Neural network components per Zhang et al. (2019) Exhibit 1."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer LSTM: 64 -> 32 units. Input (batch, seq_len, 10) -> (batch, 32)."""

    def __init__(self, n_features: int = 10, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, num_layers=1, batch_first=True)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return self.act(out[:, -1, :])  # last hidden state: (batch, 32)


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
    """Policy Gradient (REINFORCE): LSTM encoder -> raw logits."""

    def __init__(self, n_features: int = 10, n_actions: int = 3):
        super().__init__()
        self.encoder = LSTMEncoder(n_features)
        self.head = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)  # raw logits, (batch, n_actions)


class A2CNetwork(nn.Module):
    """A2C: LSTM encoder -> actor (continuous mean via tanh) + critic (value)."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.encoder = LSTMEncoder(n_features)
        self.actor = nn.Linear(32, 1)
        self.critic = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.tensor(-1.6))  # exp(-1.6) ≈ 0.2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-squash mean; action = tanh(Normal(mean, std).sample()) ∈ (-1, 1).
        # log_prob correction happens in the agent.
        h = self.encoder(x)
        mean = self.actor(h)                # (batch, 1), unbounded
        value = self.critic(h)              # (batch, 1)
        log_std = self.log_std.clamp(-5, 0)
        return mean.squeeze(-1), value.squeeze(-1), log_std
