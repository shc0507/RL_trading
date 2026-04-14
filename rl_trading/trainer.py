"""Training loop for DQN / PG / A2C agents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


class Trainer:
    """Train an RL agent on a single symbol."""

    def __init__(
        self,
        agent: DQNAgent | PGAgent | A2CAgent,
        env: TradingEnv,
        symbol: str,
        cfg: TrainerConfig | None = None,
    ):
        self.agent = agent
        self.env = env
        self.symbol = symbol
        self.cfg = cfg or TrainerConfig()
        self._ckpt_dir = Path(self.cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """Run full training loop. Returns dict with best metrics."""
        best_sharpe = -np.inf
        best_epoch = 0
        wait = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_metrics = self._run_episode("train", train=True)

            if epoch % self.cfg.eval_every == 0:
                val_metrics = self._run_episode("val", train=False)
                val_sharpe = val_metrics["sharpe"]

                print(
                    f"Epoch {epoch:3d} | "
                    f"train reward={train_metrics['total_reward']:+.4f} sharpe={train_metrics['sharpe']:+.3f} | "
                    f"val reward={val_metrics['total_reward']:+.4f} sharpe={val_metrics['sharpe']:+.3f}"
                )

                if val_sharpe > best_sharpe:
                    best_sharpe = val_sharpe
                    best_epoch = epoch
                    wait = 0
                    self.agent.save(self._ckpt_dir / f"{self.symbol}_best.pt")
                else:
                    wait += 1
                    if wait >= self.cfg.patience:
                        print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                        break

        return {"best_epoch": best_epoch, "best_val_sharpe": best_sharpe}

    def _run_episode(self, split: str, train: bool = True) -> dict:
        """Run one full episode on the given split."""
        state = self.env.reset(self.symbol, split)
        rewards: list[float] = []
        losses: list[float] = []
        done = False

        # For A2C: need to track last state for terminal handling
        while not done:
            if isinstance(self.agent, A2CAgent):
                action = self.agent.select_action(state, training=train)
            elif isinstance(self.agent, (DQNAgent, PGAgent)):
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

        # Flush remaining A2C buffer at episode end
        if train and isinstance(self.agent, A2CAgent) and len(self.agent.states) > 0:
            # Force train with whatever is left
            orig_bs = self.agent.batch_size
            self.agent.batch_size = 1
            loss = self.agent.train_step()
            self.agent.batch_size = orig_bs
            if loss is not None:
                losses.append(loss)

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
