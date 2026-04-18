"""Training loop for DQN / PG / A2C agents."""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_trading.agents import A2CAgent, DQNAgent, PGAgent
from rl_trading.env import TradingEnv


@dataclass
class TrainerConfig:
    n_epochs: int = 200
    patience: int = 20
    eval_every: int = 1
    checkpoint_dir: str = "checkpoints"
    train_dates: tuple[str, str] | None = None  # override default split dates
    val_dates: tuple[str, str] | None = None


class Trainer:
    """Train an RL agent on one or more symbols."""

    def __init__(
        self,
        agent: DQNAgent | PGAgent | A2CAgent,
        env: TradingEnv,
        symbols: list[str] | str,
        cfg: TrainerConfig | None = None,
        label: str | None = None,
    ):
        self.agent = agent
        self.env = env
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = list(symbols)
        self.label = label or self.symbols[0]
        self.cfg = cfg or TrainerConfig()
        self._ckpt_dir = Path(self.cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """Run full training loop. Returns dict with best metrics."""
        best_sharpe = -np.inf
        best_epoch = 0
        wait = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            # Train one episode per symbol (shuffled to avoid order effects)
            train_syms = list(self.symbols)
            _random.shuffle(train_syms)
            train_rewards = []
            for sym in train_syms:
                m = self._run_episode(sym, "train", train=True)
                train_rewards.append(m["total_reward"])

            if epoch % self.cfg.eval_every == 0:
                val_sharpes = []
                for sym in self.symbols:
                    m = self._run_episode(sym, "val", train=False)
                    val_sharpes.append(m["sharpe"])
                val_sharpe = float(np.mean(val_sharpes))

                print(
                    f"Epoch {epoch:3d} | "
                    f"train reward={np.mean(train_rewards):+.4f} | "
                    f"val sharpe={val_sharpe:+.3f}"
                )

                if val_sharpe > best_sharpe:
                    best_sharpe = val_sharpe
                    best_epoch = epoch
                    wait = 0
                    self.agent.save(self._ckpt_dir / f"{self.label}_best.pt")
                else:
                    wait += 1
                    if wait >= self.cfg.patience:
                        print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                        break

        return {"best_epoch": best_epoch, "best_val_sharpe": best_sharpe}

    def _run_episode(self, symbol: str, split: str, train: bool = True) -> dict:
        """Run one full episode on the given split."""
        # Use custom dates if configured, otherwise fall back to named split
        if split == "train" and self.cfg.train_dates:
            state = self.env.reset(symbol, start=self.cfg.train_dates[0],
                                   end=self.cfg.train_dates[1])
        elif split == "val" and self.cfg.val_dates:
            state = self.env.reset(symbol, start=self.cfg.val_dates[0],
                                   end=self.cfg.val_dates[1])
        else:
            state = self.env.reset(symbol, split)

        rewards: list[float] = []
        losses: list[float] = []
        done = False

        while not done:
            action = self.agent.select_action(state, training=train)

            next_state, reward, done, info = self.env.step(action)
            rewards.append(reward)

            if train:
                if isinstance(self.agent, DQNAgent):
                    ns = next_state if next_state is not None else state
                    self.agent.store(state, action, reward, ns, done)
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)

                elif isinstance(self.agent, PGAgent):
                    self.agent.store(state, action, reward)

                elif isinstance(self.agent, A2CAgent):
                    ns = next_state if next_state is not None else state
                    self.agent.store(state, action, reward, ns, done)
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)

            if next_state is not None:
                state = next_state

        # End-of-episode updates for PG
        if train and isinstance(self.agent, PGAgent):
            loss = self.agent.train_episode()
            losses.append(loss)

        # Drop leftover A2C buffer at episode end
        if train and isinstance(self.agent, A2CAgent) and len(self.agent.states) > 0:
            self.agent.states.clear()
            self.agent.actions.clear()
            self.agent.rewards.clear()
            self.agent.next_states.clear()
            self.agent.dones.clear()

        return {
            "total_reward": sum(rewards),
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "sharpe": self._compute_sharpe(rewards),
            "mean_loss": np.mean(losses) if losses else 0.0,
            "n_steps": len(rewards),
        }

    @staticmethod
    def _compute_sharpe(rewards: list[float]) -> float:
        if len(rewards) < 2:
            return 0.0
        arr = np.array(rewards)
        std = arr.std()
        if std < 1e-10:
            return 0.0
        return float(arr.mean() / std * math.sqrt(252))
