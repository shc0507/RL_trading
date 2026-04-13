"""TD3/SAC training orchestration for the continuous trading environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from ..backtest import Backtester
from ..config import DEFAULT_COST_RATE_BP, DEFAULT_OUTPUT_DIR, DEFAULT_SPLITS, DEFAULT_SYMBOLS, DEFAULT_VOL_TARGET
from ..env import EnvironmentConfig, TradingEnv
from ..policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy
from .continuous_control import ContinuousReplayBuffer, SACAgent, TD3Agent
from .env_adapter import ContinuousActorPolicy, ContinuousTradingEnv
from .trainer import TrainingArtifacts, load_or_build_features
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation


@dataclass(slots=True)
class ContinuousConfig:
    algorithm: str = "sac"
    policy_name: str | None = None
    total_steps: int = 20_000
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 2_000
    train_every: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_noise: float = 0.1
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    alpha: float = 0.2
    validation_interval: int = 2_000
    reward_mode: str = "zhang"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seed: int = 7
    device: str = "auto"
    checkpoint_metric: str = "sharpe"


class ContinuousTrainer:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        config: ContinuousConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.config = config or ContinuousConfig()
        self.config.algorithm = self.config.algorithm.lower()
        if self.config.algorithm not in {"td3", "sac"}:
            raise ValueError("ContinuousTrainer supports algorithm='td3' or algorithm='sac'")
        self.splits = splits or DEFAULT_SPLITS
        self.symbols = list(symbols or sorted(self.feature_frame["symbol"].unique().tolist()))

        self.output_dir = Path(output_dir)
        self.artifact_dir = self.output_dir / "rl"
        self.checkpoint_dir = self.artifact_dir / "checkpoints"
        self.report_dir = self.artifact_dir / "reports"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.env_config = EnvironmentConfig(
            action_mode="continuous",
            reward_mode=self.config.reward_mode,
            cost_rate_bp=self.config.cost_rate_bp,
            vol_target=self.config.vol_target,
        )
        self.rl_env = ContinuousTradingEnv(
            feature_frame=self.feature_frame,
            config=self.env_config,
            splits=self.splits,
        )
        self.backtester = Backtester(
            feature_frame=self.feature_frame,
            env_config=self.env_config,
            splits=self.splits,
        )
        self.symbols_by_split = {
            split_name: self._resolve_valid_symbols(split_name)
            for split_name in self.splits
        }
        if not self.symbols_by_split.get("train"):
            raise ValueError("no valid symbols are available for the train split")

        agent_kwargs = {
            "state_size": self.rl_env.state_size,
            "action_size": self.rl_env.action_size,
            "hidden_sizes": self.config.hidden_sizes,
            "activation": self.config.activation,
            "actor_lr": self.config.actor_lr,
            "critic_lr": self.config.critic_lr,
            "gamma": self.config.gamma,
            "tau": self.config.tau,
            "device": self.config.device,
        }
        if self.config.algorithm == "td3":
            self.agent = TD3Agent(
                **agent_kwargs,
                policy_delay=self.config.policy_delay,
                target_noise=self.config.target_noise,
                target_noise_clip=self.config.target_noise_clip,
            )
        else:
            self.agent = SACAgent(**agent_kwargs, alpha=self.config.alpha)
        self.buffer = ContinuousReplayBuffer(
            capacity=self.config.replay_capacity,
            state_size=self.rl_env.state_size,
            action_size=self.rl_env.action_size,
            device=str(self.agent.device),
        )
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.final_checkpoint_path = self.checkpoint_dir / "final_model.pt"
        self._seed_everything()

    def _seed_everything(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

    def _resolve_valid_symbols(self, split: str) -> list[str]:
        validation_env = TradingEnv(
            feature_frame=self.feature_frame,
            config=self.env_config,
            splits=self.splits,
        )
        valid_symbols: list[str] = []
        for symbol in self.symbols:
            try:
                validation_env.reset(symbol=symbol, split=split)
            except ValueError:
                continue
            valid_symbols.append(symbol)
        return valid_symbols

    def _sample_symbol(self, split: str) -> str:
        candidates = self.symbols_by_split[split]
        return candidates[int(self.rng.integers(len(candidates)))]

    def _write_frame_csv_atomic(self, path: Path, frame: pd.DataFrame) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _policy_name(self) -> str:
        return self.config.policy_name or self.config.algorithm

    def _current_policy(self) -> ContinuousActorPolicy:
        return ContinuousActorPolicy(
            agent=self.agent,
            observation_window=self.rl_env.observation_window,
            feature_columns=self.rl_env.feature_columns,
            name=self._policy_name(),
        )

    def _evaluate_split(self, policy, split: str):
        symbols = self.symbols_by_split.get(split, [])
        if not symbols:
            return None
        return self.backtester.run(policy=policy, split=split, symbols=symbols)

    def _persist_config(self) -> Path:
        config_path = self.artifact_dir / "config.json"
        payload = {
            "trainer": asdict(self.config),
            "splits": self.splits,
            "symbols": self.symbols,
            "symbols_by_split": self.symbols_by_split,
            "state_size": self.rl_env.state_size,
            "action_range": [-1.0, 1.0],
        }
        self._write_json_atomic(config_path, payload)
        return config_path

    def _save_report_files(self, report) -> None:
        prefix = f"{report.policy_name}_{report.split}"
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_daily_returns.csv", report.daily_returns)
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_trades.csv", report.trade_log)
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_symbol_metrics.csv", report.symbol_metrics)

    def _select_training_action(self, state: np.ndarray, step: int) -> float:
        if step <= self.config.warmup_steps:
            return float(self.rng.uniform(-1.0, 1.0))
        if self.config.algorithm == "td3":
            return float(self.agent.get_action(state, noise_std=self.config.exploration_noise))
        return float(self.agent.get_action(state, deterministic=False))

    def train(self) -> TrainingArtifacts:
        config_path = self._persist_config()
        progress_rows: list[dict[str, object]] = []
        best_metric = float("-inf")
        has_validation = False
        state, _ = self.rl_env.reset(symbol=self._sample_symbol("train"), split="train")
        episode_return = 0.0
        episode_length = 0

        for step in range(1, self.config.total_steps + 1):
            action = self._select_training_action(state, step)
            next_state, reward, terminated, truncated, _ = self.rl_env.step(action)
            done = terminated or truncated
            self.buffer.add((state, np.array([action], dtype=np.float32), reward, next_state, done))
            state = next_state
            episode_return += float(reward)
            episode_length += 1

            if (
                step >= self.config.warmup_steps
                and len(self.buffer) >= self.config.batch_size
                and step % self.config.train_every == 0
            ):
                update_info = self.agent.update(self.buffer.sample(self.config.batch_size), step=step)
                progress_rows.append(
                    {
                        "event": "update",
                        "step": step,
                        "critic_loss": update_info.get("critic_loss", np.nan),
                        "actor_loss": update_info.get("actor_loss", np.nan),
                        "entropy": update_info.get("entropy", np.nan),
                        "episode_return": np.nan,
                        "episode_length": np.nan,
                        "validation_metric": np.nan,
                    }
                )

            if done:
                progress_rows.append(
                    {
                        "event": "episode",
                        "step": step,
                        "critic_loss": np.nan,
                        "actor_loss": np.nan,
                        "entropy": np.nan,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "validation_metric": np.nan,
                    }
                )
                state, _ = self.rl_env.reset(symbol=self._sample_symbol("train"), split="train")
                episode_return = 0.0
                episode_length = 0

            if (
                self.config.validation_interval > 0
                and step % self.config.validation_interval == 0
                and self.symbols_by_split.get("val")
            ):
                report = self._evaluate_split(self._current_policy(), split="val")
                if report is not None:
                    has_validation = True
                    metric = float(report.portfolio_metrics[self.config.checkpoint_metric])
                    progress_rows.append(
                        {
                            "event": "validation",
                            "step": step,
                            "critic_loss": np.nan,
                            "actor_loss": np.nan,
                            "entropy": np.nan,
                            "episode_return": np.nan,
                            "episode_length": np.nan,
                            "validation_metric": metric,
                        }
                    )
                    if metric > best_metric:
                        best_metric = metric
                        self.agent.save(self.best_checkpoint_path)

        self.agent.save(self.final_checkpoint_path)
        if not has_validation:
            self.agent.save(self.best_checkpoint_path)
        self.agent.load(self.best_checkpoint_path)

        policies = [self._current_policy(), LongOnlyPolicy(), Sign12MPolicy(), MACDPolicy()]
        summary_rows: list[dict[str, object]] = []
        for split in self.splits:
            if not self.symbols_by_split.get(split):
                continue
            for policy in policies:
                report = self._evaluate_split(policy=policy, split=split)
                if report is None:
                    continue
                summary_rows.append({"policy": report.policy_name, "split": split, **report.portfolio_metrics})
                self._save_report_files(report)

        training_curve = pd.DataFrame(progress_rows)
        summary_metrics = pd.DataFrame(summary_rows)
        if not summary_metrics.empty:
            summary_metrics = summary_metrics.sort_values(["split", "policy"]).reset_index(drop=True)
        training_curve_path = self.artifact_dir / "training_curve.csv"
        summary_metrics_path = self.artifact_dir / "summary_metrics.csv"
        self._write_frame_csv_atomic(training_curve_path, training_curve)
        self._write_frame_csv_atomic(summary_metrics_path, summary_metrics)
        zhang_eval_artifacts = run_zhang_evaluation(
            artifact_dir=self.artifact_dir,
            config=ZhangEvalConfig(
                target_vol=self.config.vol_target,
                split=None,
                return_column="raw_return" if self.env_config.reward_mode == "raw" else "zhang_return",
            ),
        )
        return TrainingArtifacts(
            artifact_dir=self.artifact_dir,
            config_path=config_path,
            training_curve_path=training_curve_path,
            summary_metrics_path=summary_metrics_path,
            best_checkpoint_path=self.best_checkpoint_path,
            final_checkpoint_path=self.final_checkpoint_path,
            zhang_eval_dir=zhang_eval_artifacts.output_dir,
        )


def train_continuous(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: ContinuousConfig | None = None,
    feature_frame: pd.DataFrame | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    rebuild_data: bool = False,
) -> TrainingArtifacts:
    features = feature_frame if feature_frame is not None else load_or_build_features(
        output_dir=output_dir,
        symbols=tuple(symbols) if symbols is not None else DEFAULT_SYMBOLS,
        rebuild_data=rebuild_data,
    )
    trainer = ContinuousTrainer(
        feature_frame=features,
        output_dir=output_dir,
        config=config,
        splits=splits,
        symbols=symbols,
    )
    return trainer.train()
