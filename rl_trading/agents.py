"""RL agents per Zhang et al. (2019): DQN, PG (REINFORCE), A2C."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_trading.networks import A2CNetwork, DQNNetwork, PGNetwork


# ── DQN Agent ──────────────────────────────────────────────────────────

class DQNAgent:
    """Double DQN with dueling architecture and experience replay."""

    def __init__(
        self,
        n_features: int = 10,
        lr: float = 1e-4,
        gamma: float = 0.3,
        batch_size: int = 64,
        memory_size: int = 5000,
        target_update_freq: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 5000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.online = DQNNetwork(n_features).to(self.device)
        self.target = DQNNetwork(n_features).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.memory: deque = deque(maxlen=memory_size)
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        frac = min(self.step_count / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(t)
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Double DQN: online selects action, target evaluates
        q_vals = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.online(ns).argmax(dim=1)
            q_next = self.target(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target()

        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path):
        torch.save(self.online.state_dict(), path)

    def load(self, path: str | Path):
        self.online.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target.load_state_dict(self.online.state_dict())


# ── PG Agent (REINFORCE) ──────────────────────────────────────────────

class PGAgent:
    """REINFORCE with reward-to-go and mean baseline."""

    def __init__(
        self,
        n_features: int = 10,
        lr: float = 1e-4,
        gamma: float = 0.3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma

        self.net = PGNetwork(n_features).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Episode buffers
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.net(t).squeeze(0)
        if training:
            action = torch.multinomial(probs, 1).item()
        else:
            action = int(probs.argmax().item())
        return action

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train_episode(self) -> float:
        """Update policy after a full episode. Returns loss value."""
        if not self.rewards:
            return 0.0

        # Discounted reward-to-go
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Baseline: subtract mean
        returns = returns - returns.mean()
        if returns.std() > 1e-8:
            returns = returns / returns.std()

        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a = torch.tensor(self.actions, dtype=torch.long, device=self.device)

        probs = self.net(s)
        log_probs = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-8)
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return loss.item()

    def save(self, path: str | Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


# ── A2C Agent (continuous) ────────────────────────────────────────────

class A2CAgent:
    """Advantage Actor-Critic with continuous actions and TD advantage."""

    def __init__(
        self,
        n_features: int = 10,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-4,
        batch_size: int = 128,
        gamma: float = 0.3,
        std: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.std = std

        self.net = A2CNetwork(n_features).to(self.device)
        self.opt_actor = optim.Adam(
            list(self.net.encoder.parameters()) + list(self.net.actor.parameters()),
            lr=lr_actor,
        )
        self.opt_critic = optim.Adam(
            list(self.net.encoder.parameters()) + list(self.net.critic.parameters()),
            lr=lr_critic,
        )

        # Step buffer
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.next_states: list = []
        self.dones: list = []

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _ = self.net(t)
            mean = mean.item()
        if training:
            action = np.clip(np.random.normal(mean, self.std), -1.0, 1.0)
        else:
            action = mean
        return float(action)

    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def train_step(self) -> float | None:
        if len(self.states) < self.batch_size:
            return None

        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        r = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(self.next_states), dtype=torch.float32, device=self.device)
        d = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        mean, value = self.net(s)
        with torch.no_grad():
            _, next_value = self.net(ns)
            td_target = r + self.gamma * next_value * (1 - d)
        advantage = (td_target - value).detach()

        # Critic loss
        critic_loss = nn.functional.mse_loss(value, td_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Actor loss (Gaussian log prob)
        mean, _ = self.net(s)  # recompute after critic update
        dist = torch.distributions.Normal(mean, self.std)
        log_prob = dist.log_prob(a)
        actor_loss = -(log_prob * advantage).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        return (actor_loss.item() + critic_loss.item()) / 2

    def save(self, path: str | Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
