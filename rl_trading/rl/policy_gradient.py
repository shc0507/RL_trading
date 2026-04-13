"""Policy-gradient and actor-critic agents for discrete trading actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from .torch_utils import as_float_tensor, build_mlp, resolve_device


@dataclass(slots=True)
class OnPolicyBatch:
    states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor | None = None


def discounted_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    total = 0.0
    for index, reward in enumerate(rewards):
        total += float(reward) * (float(gamma) ** index)
    return np.full_like(rewards, total, dtype=np.float32)


def discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + float(gamma) * running
        values[index] = running
    return values


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)
    values = values.reshape(-1)
    next_value = next_value.reshape(())
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(rewards.shape[0])):
        following_value = next_value if index == rewards.shape[0] - 1 else values[index + 1]
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * following_value * non_terminal - values[index]
        running_advantage = delta + gamma * gae_lambda * non_terminal * running_advantage
        advantages[index] = running_advantage
    returns = advantages + values
    return advantages, returns


class CategoricalActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.logits_net = build_mlp(
            input_size=state_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> distributions.Categorical:
        return distributions.Categorical(logits=self.logits_net(states))


class ValueCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.value_net = build_mlp(
            input_size=state_size,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)


class A2CAgent:
    """Advantage actor-critic agent for the existing discrete action adapter."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float | None = 0.5,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = max_grad_norm
        self.actor = CategoricalActor(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.critic = ValueCritic(
            state_size=state_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def get_action(self, state, deterministic: bool = False) -> tuple[int, float, float]:
        state_tensor = as_float_tensor(state, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        distribution = self.actor(state_tensor)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        value = self.critic(state_tensor)
        log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_greedy_action(self, state) -> int:
        action, _, _ = self.get_action(state, deterministic=True)
        return action

    def _policy_loss(self, distribution, actions, advantages, old_log_probs=None):
        del old_log_probs
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        actor_loss = -(log_probs * advantages).mean()
        return actor_loss, entropy, log_probs

    def update(self, batch: OnPolicyBatch) -> dict[str, float]:
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device).long()
        returns = batch.returns.to(self.device).float()
        advantages = batch.advantages.to(self.device).float()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        distribution = self.actor(states)
        values = self.critic(states)
        actor_loss, entropy, _ = self._policy_loss(distribution, actions, advantages, batch.old_log_probs)
        value_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            target,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


class PPOAgent(A2CAgent):
    """Clipped PPO agent for discrete trading actions."""

    def __init__(
        self,
        *args,
        clip_ratio: float = 0.2,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_ratio = float(clip_ratio)
        self.update_epochs = max(1, int(update_epochs))
        self.minibatch_size = max(1, int(minibatch_size))

    def _ppo_losses(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.actor(states)
        log_probs = distribution.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        actor_loss = -torch.min(unclipped, clipped).mean()
        values = self.critic(states)
        value_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss, actor_loss, value_loss, entropy

    def update(self, batch: OnPolicyBatch) -> dict[str, float]:
        if batch.old_log_probs is None:
            raise ValueError("PPOAgent.update requires old_log_probs")
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device).long()
        returns = batch.returns.to(self.device).float()
        advantages = batch.advantages.to(self.device).float()
        old_log_probs = batch.old_log_probs.to(self.device).float()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        last_info: dict[str, float] = {}
        indices = torch.arange(states.shape[0], device=self.device)
        for _ in range(self.update_epochs):
            shuffled = indices[torch.randperm(indices.shape[0], device=self.device)]
            for start in range(0, states.shape[0], self.minibatch_size):
                batch_indices = shuffled[start : start + self.minibatch_size]
                total_loss, actor_loss, value_loss, entropy = self._ppo_losses(
                    states[batch_indices],
                    actions[batch_indices],
                    returns[batch_indices],
                    advantages[batch_indices],
                    old_log_probs[batch_indices],
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                last_info = {
                    "loss": float(total_loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_info
