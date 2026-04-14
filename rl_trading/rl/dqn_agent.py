"""DQN-family agents adapted from the homework implementation."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from .dqn_model import build_q_network
from .replay_buffer import TransitionBatch
from .torch_utils import as_float_tensor, resolve_device


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_update_interval: int = 1_000,
        device: str = "auto",
        double_dqn: bool = False,
        dueling: bool = False,
        network_type: str = "mlp",
        observation_window: int | None = None,
        feature_size: int | None = None,
        recurrent_hidden_size: int = 128,
        recurrent_layers: int = 2,
        recurrent_layer_sizes: Sequence[int] | None = None,
        recurrent_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        self.device = resolve_device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = max(1, int(target_update_interval))
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
        self.network_type = network_type

        self.q_net = build_q_network(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            network_type=network_type,
            dueling=dueling,
            observation_window=observation_window,
            feature_size=feature_size,
            recurrent_hidden_size=recurrent_hidden_size,
            recurrent_layers=recurrent_layers,
            recurrent_layer_sizes=recurrent_layer_sizes,
            recurrent_dropout=recurrent_dropout,
            head_dropout=head_dropout,
        ).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * source_param.data)

    @torch.no_grad()
    def get_action(self, state) -> int:
        state_tensor = as_float_tensor(state, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        q_values = self.q_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def get_q_target(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.double_dqn:
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(-1)
        else:
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.max(dim=1).values
        return rewards + (1.0 - dones) * self.gamma * next_q

    def get_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_values = self.q_net(states)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(-1)

    def update(self, batch: TransitionBatch, step: int) -> dict[str, object]:
        q_target = self.get_q_target(
            rewards=batch.rewards,
            dones=batch.dones,
            next_states=batch.next_states,
        )
        q_values = self.get_q(batch.states, batch.actions)
        td_error = (q_values - q_target).abs().detach()

        loss = F.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.target_update_interval == 0:
            with torch.no_grad():
                self.soft_update(self.target_net, self.q_net)

        return {
            "loss": float(loss.item()),
            "td_error": td_error,
            "mean_q": float(q_values.mean().item()),
            "mean_target": float(q_target.mean().item()),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), target)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(Path(path), map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def __repr__(self) -> str:
        pieces = []
        if self.double_dqn:
            pieces.append("Double")
        if self.dueling:
            pieces.append("Dueling")
        if self.network_type.lower() == "lstm":
            pieces.append("LSTM")
        pieces.append("DQNAgent")
        return "".join(pieces)
