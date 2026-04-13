"""Continuous-action off-policy agents for target-position trading."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from .torch_utils import as_float_tensor, build_mlp, resolve_device


@dataclass(slots=True)
class ContinuousTransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ContinuousReplayBuffer:
    def __init__(self, capacity: int, state_size: int, action_size: int, device: str = "auto") -> None:
        self.capacity = max(1, int(capacity))
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = resolve_device(device)
        self.states = torch.empty((self.capacity, self.state_size), dtype=torch.float32)
        self.actions = torch.empty((self.capacity, self.action_size), dtype=torch.float32)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.next_states = torch.empty((self.capacity, self.state_size), dtype=torch.float32)
        self.dones = torch.empty(self.capacity, dtype=torch.float32)
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: tuple[np.ndarray, np.ndarray | float, float, np.ndarray, bool | float]) -> None:
        state, action, reward, next_state, done = transition
        self.states[self.index] = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.float32).reshape(self.action_size)
        self.rewards[self.index] = float(reward)
        self.next_states[self.index] = torch.as_tensor(next_state, dtype=torch.float32).reshape(-1)
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> ContinuousTransitionBatch:
        if batch_size > self.size:
            raise ValueError(f"cannot sample batch_size={batch_size} from buffer of size={self.size}")
        indices = np.random.choice(self.size, batch_size, replace=False)
        return ContinuousTransitionBatch(
            states=self.states[indices].to(self.device),
            actions=self.actions[indices].to(self.device),
            rewards=self.rewards[indices].to(self.device),
            next_states=self.next_states[indices].to(self.device),
            dones=self.dones[indices].to(self.device),
        )


class DeterministicActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            input_size=state_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(states))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
    ) -> None:
        super().__init__()
        self.action_size = int(action_size)
        self.log_std_bounds = log_std_bounds
        self.net = build_mlp(
            input_size=state_size,
            output_size=2 * action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(states).chunk(2, dim=-1)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(states)
        normal = distributions.Normal(mean, log_std.exp())
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    @torch.no_grad()
    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(states)
        return torch.tanh(mean)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_size = int(state_size) + int(action_size)
        self.q1 = build_mlp(input_size, 1, hidden_sizes, activation=activation)
        self.q2 = build_mlp(input_size, 1, hidden_sizes, activation=activation)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([states, actions], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def q1_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([states, actions], dim=-1)
        return self.q1(inputs).squeeze(-1)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


class TD3Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = max(1, int(policy_delay))
        self.target_noise = float(target_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.actor = DeterministicActor(state_size, action_size, hidden_sizes, activation).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic = TwinQCritic(state_size, action_size, hidden_sizes, activation).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def get_action(self, state, noise_std: float = 0.0) -> float:
        state_tensor = as_float_tensor(state, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action = self.actor(state_tensor)
        if noise_std > 0.0:
            action = action + torch.randn_like(action) * float(noise_std)
        action = action.clamp(-1.0, 1.0)
        return float(action.squeeze(0).squeeze(-1).item())

    def update(self, batch: ContinuousTransitionBatch, step: int) -> dict[str, float]:
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            q_target = rewards + (1.0 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info = {"critic_loss": float(critic_loss.item())}
        if step % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_value(states, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            with torch.no_grad():
                _soft_update(self.actor_target, self.actor, self.tau)
                _soft_update(self.critic_target, self.critic, self.tau)
            info["actor_loss"] = float(actor_loss.item())
        return info

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            target,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])


class SACAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.actor = SquashedGaussianActor(state_size, action_size, hidden_sizes, activation).to(self.device)
        self.critic = TwinQCritic(state_size, action_size, hidden_sizes, activation).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def get_action(self, state, deterministic: bool = False) -> float:
        state_tensor = as_float_tensor(state, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(state_tensor)
        else:
            action, _ = self.actor.sample(state_tensor)
        return float(action.squeeze(0).squeeze(-1).clamp(-1.0, 1.0).item())

    def update(self, batch: ContinuousTransitionBatch, step: int | None = None) -> dict[str, float]:
        del step
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha * next_log_probs
            q_target = rewards + (1.0 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        actor_loss = (self.alpha * log_probs - torch.minimum(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            _soft_update(self.critic_target, self.critic, self.tau)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float((-log_probs).mean().item()),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            target,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
