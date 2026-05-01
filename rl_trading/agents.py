"""RL agents per Zhang et al. (2019): DQN, PG (REINFORCE), A2C."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.grad_step_count = 0  # gradient updates; drives target-net refresh
        self.env_step_count = 0   # env interactions during training; drives ε schedule
        self.last_stats: dict[str, float] = {}

    @property
    def epsilon(self) -> float:
        frac = min(self.env_step_count / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            self.env_step_count += 1
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        self.online.eval()
        try:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.online(t)
                return int(q.argmax(dim=1).item())
        finally:
            if training:
                self.online.train()

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

        # Switch to train mode before the gradient forward. select_action leaves
        # online in eval, and cuDNN RNN requires train mode for backward.
        self.online.train()

        # Double DQN: online selects action, target evaluates
        q_vals = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Bootstrap argmax in eval mode so dropout doesn't randomize
            # the action the target network is asked to value.
            self.online.eval()
            best_actions = self.online(ns).argmax(dim=1)
            self.online.train()
            q_next = self.target(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.grad_step_count += 1
        if self.grad_step_count % self.target_update_freq == 0:
            self.update_target()

        self.last_stats = {
            "loss": loss.item(),
            "q_mean": q_vals.mean().item(),
            "td_error": (targets - q_vals).abs().mean().item(),
            "epsilon": self.epsilon,
            "replay_size": float(len(self.memory)),
        }
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
    """REINFORCE with reward-to-go, mean baseline, and entropy bonus.

    Deviates from Zhang Eq. 6 / Exhibit 1 in three ways that are essentially
    required to get REINFORCE working under 20-bp transaction costs and 50+
    daily-bar episodes; paper omits these stabilizers, but they are standard
    practice and prevent the silent-PG attractor where the softmax saturates
    at "always hold" because trading costs > episode-mean reward signal:
      - γ = 0.95 (paper Exhibit 1 says 0.3, but γ=0.3 gives effective horizon
        ~3 bars and leaves G_t too small to overcome the cost baseline).
      - entropy_coef = 0.05 (no entropy bonus in paper; without it the
        softmax collapses to one-hot within ~10 epochs).
      - Mean-of-batch baseline (unbiased; already restored).
    """

    def __init__(
        self,
        n_features: int = 10,
        # v7: lr 1e-4 -> 5e-4. PG gradients are tiny relative to A2C/DQN
        # because per-bar return signal is small after /p_ref normalization
        # and one update per episode means few opportunities to move.
        lr: float = 5e-4,
        gamma: float = 0.95,
        # v7: entropy_coef 0.05 -> 0.01. The value gradient is O(0.05) per
        # sample (|G_t| ~ 0.05 with γ=0.95); a 0.05 entropy coefficient
        # was the same magnitude and kept the softmax pinned to uniform.
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.net = PGNetwork(n_features).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Episode buffers
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.last_stats: dict[str, float] = {}

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        self.net.eval()
        try:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.net(t).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
        finally:
            if training:
                self.net.train()

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

        # Subtract the batch mean as a constant baseline. Paper Eq. 6 shows
        # vanilla REINFORCE, but with dollar-unit rewards on high-priced
        # commodity contracts the DC offset in G_t saturates the softmax
        # and training diverges (inf/nan in multinomial). Subtracting the
        # mean is unbiased — E[∇log π · b] = 0 for any state-independent b
        # — so the expected gradient matches Eq. 6; only variance is lower.
        returns = returns - returns.mean()

        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        a = torch.tensor(self.actions, dtype=torch.long, device=self.device)

        logits = self.net(s)
        log_probs_full = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs_full.exp() * log_probs_full).sum(dim=-1).mean()
        log_probs = log_probs_full.gather(1, a.unsqueeze(1)).squeeze(1)
        # Maximize expected return + entropy_coef * H(π).
        loss = -(log_probs * returns).mean() - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.last_stats = {
            "loss": loss.item(),
            "entropy": entropy.item(),
            "return_mean": float(returns.mean().item()),
            "return_std": float(returns.std().item()) if returns.numel() > 1 else 0.0,
            "episode_len": float(len(self.rewards)),
        }

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
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        batch_size: int = 128,
        gamma: float = 0.3,
        value_coef: float = 0.5,
        # v8: revert to 0.0 (paper-literal). v5 (entropy=0) achieved +0.50
        # All Sharpe; v6/v7 (entropy=0.01) regressed to -0.77. A2C's
        # TanhNormal already gets exploration from the Gaussian σ.
        entropy_coef: float = 0.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.net = A2CNetwork(n_features).to(self.device)
        self.optimizer = optim.Adam([
            {"params": self.net.encoder.parameters(), "lr": lr_actor},
            {"params": list(self.net.actor.parameters()) + [self.net.log_std],
             "lr": lr_actor},
            {"params": self.net.critic.parameters(), "lr": lr_critic},
        ])

        # Step buffer
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.next_states: list = []
        self.dones: list = []
        self.last_stats: dict[str, float] = {}

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        # TanhNormal policy: sample u ~ N(mean, std), action = tanh(u) ∈ (-1, 1).
        self.net.eval()
        try:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mean, _, log_std = self.net(t)
                mean = mean.item()
                std = log_std.exp().item()
        finally:
            if training:
                self.net.train()
        u = np.random.normal(mean, std) if training else mean
        action = float(np.tanh(u))
        return action

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

        mean, value, log_std = self.net(s)
        std = log_std.exp()
        with torch.no_grad():
            _, next_value, _ = self.net(ns)
            td_target = r + self.gamma * next_value * (1 - d)
        advantage = (td_target - value).detach()

        # Critic loss
        critic_loss = nn.functional.mse_loss(value, td_target)

        # TanhNormal log prob: action a = tanh(u), u ~ N(mean, std).
        # Recover u = atanh(a) with numerical clamp; Jacobian correction is
        # -log(1 - a^2).  See Haarnoja et al. (2018) SAC appendix C.
        a_clamped = a.clamp(-0.999999, 0.999999)
        u = torch.atanh(a_clamped)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u) - torch.log1p(-a_clamped.pow(2) + 1e-6)
        actor_loss = -(log_prob * advantage).mean()

        # Entropy bonus uses the pre-squash Gaussian entropy (common
        # TanhNormal approximation; true entropy has no closed form).
        entropy = dist.entropy().mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.last_stats = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item() if advantage.numel() > 1 else 0.0,
            "value_mean": value.mean().item(),
            "log_std": log_std.item(),
        }

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        return loss.item()

    def save(self, path: str | Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
