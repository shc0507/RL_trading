"""Training loop for DQN / PG / A2C agents."""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_trading.agents import A2CAgent, DQNAgent, PGAgent
from rl_trading.env import TradingEnv
from rl_trading.wandb_logger import WandbLogger


@dataclass
class TrainerConfig:
    n_epochs: int = 200
    patience: int = 20
    eval_every: int = 1
    checkpoint_dir: str = "checkpoints"
    train_dates: tuple[str, str] | None = None
    val_dates: tuple[str, str] | None = None
    # "sharpe_only" (paper-literal) or "or_multi" (Sharpe AND Sortino
    # AND Cum all stale; gives slow learners more training time).
    early_stop_policy: str = "sharpe_only"


class Trainer:
    """Train an RL agent on one or more symbols."""

    def __init__(
        self,
        agent: DQNAgent | PGAgent | A2CAgent,
        env: TradingEnv,
        symbols: list[str] | str,
        cfg: TrainerConfig | None = None,
        label: str | None = None,
        logger: WandbLogger | None = None,
        context: str | None = None,
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
        self.logger = logger
        self.context = context
        if self.logger is not None and self.context is not None:
            self.logger.define_context(self.context)

    def train(self) -> dict:
        """Train with configurable early stopping (see TrainerConfig).

        Checkpoint is always saved on Sharpe improvement; Sortino and Cum
        only gate stopping under the ``or_multi`` policy.
        """
        policy = self.cfg.early_stop_policy
        if policy not in ("sharpe_only", "or_multi"):
            raise ValueError(f"early_stop_policy must be sharpe_only or or_multi; got {policy}")

        best_sharpe = -np.inf
        best_sortino = -np.inf
        best_cum = -np.inf
        wait_sharpe = 0
        wait_sortino = 0
        wait_cum = 0
        best_epoch = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_syms = list(self.symbols)
            _random.shuffle(train_syms)
            train_rewards = []
            train_sharpes = []
            train_losses = []
            agent_stats_acc: dict[str, list[float]] = {}
            for sym in train_syms:
                m = self._run_episode(sym, "train", train=True)
                train_rewards.append(m["total_reward"])
                train_sharpes.append(m["sharpe"])
                train_losses.append(m["mean_loss"])
                for k, v in m.get("agent_stats", {}).items():
                    agent_stats_acc.setdefault(k, []).append(v)

            if epoch % self.cfg.eval_every == 0:
                val_sharpes = []
                val_sortinos = []
                val_cum_returns = []
                for sym in self.symbols:
                    m = self._run_episode(sym, "val", train=False)
                    val_sharpes.append(m["sharpe"])
                    val_sortinos.append(m["sortino"])
                    val_cum_returns.append(m["total_reward"])
                cur_sharpe = float(np.mean(val_sharpes))
                cur_sortino = float(np.mean(val_sortinos))
                cur_cum = float(np.mean(val_cum_returns))

                if cur_sharpe > best_sharpe:
                    best_sharpe = cur_sharpe
                    best_epoch = epoch
                    wait_sharpe = 0
                    self.agent.save(self._ckpt_dir / f"{self.label}_best.pt")
                else:
                    wait_sharpe += 1
                if cur_sortino > best_sortino:
                    best_sortino = cur_sortino
                    wait_sortino = 0
                else:
                    wait_sortino += 1
                if cur_cum > best_cum:
                    best_cum = cur_cum
                    wait_cum = 0
                else:
                    wait_cum += 1

                if policy == "sharpe_only":
                    stop_now = wait_sharpe >= self.cfg.patience
                    wait_str = f"{wait_sharpe:2d}/{self.cfg.patience}"
                else:  # or_multi
                    stop_now = (
                        wait_sharpe >= self.cfg.patience
                        and wait_sortino >= self.cfg.patience
                        and wait_cum >= self.cfg.patience
                    )
                    wait_str = (
                        f"S={wait_sharpe:2d}/So={wait_sortino:2d}/"
                        f"C={wait_cum:2d} (need all >= {self.cfg.patience})"
                    )

                print(
                    f"Epoch {epoch:3d} | "
                    f"train reward={np.mean(train_rewards):+.4f} | "
                    f"val sharpe={cur_sharpe:+.3f} "
                    f"sortino={cur_sortino:+.3f} "
                    f"cum={cur_cum:+.3f} | "
                    f"wait={wait_str}"
                )

                if self.logger is not None and self.context is not None:
                    payload: dict[str, float] = {
                        "train/reward_mean": float(np.mean(train_rewards)),
                        "train/reward_sum": float(np.sum(train_rewards)),
                        "train/sharpe": float(np.mean(train_sharpes)),
                        "train/loss": float(np.mean(train_losses)),
                        "val/sharpe": cur_sharpe,
                        "val/sortino": cur_sortino,
                        "val/cum_return": cur_cum,
                        "best/val_sharpe": float(best_sharpe),
                        "best/val_sortino": float(best_sortino),
                        "best/val_cum_return": float(best_cum),
                        "best/epoch": float(best_epoch),
                        "early_stop/wait_sharpe": float(wait_sharpe),
                        "early_stop/wait_sortino": float(wait_sortino),
                        "early_stop/wait_cum_return": float(wait_cum),
                    }
                    for k, vs in agent_stats_acc.items():
                        if vs:
                            payload[f"agent/{k}"] = float(np.mean(vs))
                    self.logger.log(payload, ctx=self.context, epoch=epoch)

                if stop_now:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(policy={policy}, patience={self.cfg.patience}; "
                        f"Sharpe-best epoch={best_epoch})"
                    )
                    break

        return {
            "best_epoch": best_epoch,
            "best_val_sharpe": best_sharpe,
            "best_val_sortino": best_sortino,
            "best_val_cum_return": best_cum,
        }

    def _run_episode(self, symbol: str, split: str, train: bool = True) -> dict:
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
        stats_snapshots: list[dict[str, float]] = []
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
                        stats_snapshots.append(dict(self.agent.last_stats))

                elif isinstance(self.agent, PGAgent):
                    self.agent.store(state, action, reward)

                elif isinstance(self.agent, A2CAgent):
                    ns = next_state if next_state is not None else state
                    self.agent.store(state, action, reward, ns, done)
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                        stats_snapshots.append(dict(self.agent.last_stats))

            if next_state is not None:
                state = next_state

        if train and isinstance(self.agent, PGAgent):
            loss = self.agent.train_episode()
            losses.append(loss)
            stats_snapshots.append(dict(self.agent.last_stats))

        if train and isinstance(self.agent, A2CAgent) and len(self.agent.states) > 0:
            self.agent.states.clear()
            self.agent.actions.clear()
            self.agent.rewards.clear()
            self.agent.next_states.clear()
            self.agent.dones.clear()

        agent_stats: dict[str, float] = {}
        if stats_snapshots:
            keys = stats_snapshots[0].keys()
            for k in keys:
                agent_stats[k] = float(np.mean([s[k] for s in stats_snapshots if k in s]))

        return {
            "total_reward": sum(rewards),
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "sharpe": self._compute_sharpe(rewards),
            "sortino": self._compute_sortino(rewards),
            "mean_loss": np.mean(losses) if losses else 0.0,
            "n_steps": len(rewards),
            "agent_stats": agent_stats,
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

    @staticmethod
    def _compute_sortino(rewards: list[float]) -> float:
        """Annualized Sortino: mean / downside_std (only negative returns).

        Falls back to Sharpe when there isn't enough downside variance
        to estimate — keeps the metric comparable across epochs instead
        of collapsing to 0 or exploding.
        """
        if len(rewards) < 2:
            return 0.0
        arr = np.array(rewards)
        downside = arr[arr < 0]
        ds_std = downside.std() if downside.size >= 2 else 0.0
        if ds_std < 1e-10:
            return Trainer._compute_sharpe(rewards)
        return float(arr.mean() / ds_std * math.sqrt(252))
