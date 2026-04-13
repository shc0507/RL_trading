"""Replay buffer adapted from the homework DQN buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .torch_utils import resolve_device


@dataclass(slots=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, device: str = "auto") -> None:
        self.capacity = max(1, int(capacity))
        self.state_size = int(state_size)
        self.device = resolve_device(device)

        self.states = torch.empty((self.capacity, self.state_size), dtype=torch.float32)
        self.actions = torch.empty(self.capacity, dtype=torch.int64)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.next_states = torch.empty((self.capacity, self.state_size), dtype=torch.float32)
        self.dones = torch.empty(self.capacity, dtype=torch.float32)

        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        transition: tuple[np.ndarray | torch.Tensor, int, float, np.ndarray | torch.Tensor, bool | float],
    ) -> None:
        state, action, reward, next_state, done = transition
        self.states[self.index] = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        self.actions[self.index] = int(action)
        self.rewards[self.index] = float(reward)
        self.next_states[self.index] = torch.as_tensor(next_state, dtype=torch.float32).reshape(-1)
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size > self.size:
            raise ValueError(f"cannot sample batch_size={batch_size} from buffer of size={self.size}")

        sample_indices = np.random.choice(self.size, batch_size, replace=False)
        return TransitionBatch(
            states=self.states[sample_indices].to(self.device),
            actions=self.actions[sample_indices].to(self.device),
            rewards=self.rewards[sample_indices].to(self.device),
            next_states=self.next_states[sample_indices].to(self.device),
            dones=self.dones[sample_indices].to(self.device),
        )
