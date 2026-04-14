"""DQN training orchestration for the trading environment."""

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
from ..config import (
    DEFAULT_COST_RATE_BP,
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLITS,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOLS,
    DEFAULT_VOL_TARGET,
)
from ..data.pipeline import MarketDataPipeline
from ..data.sources import PublicDailySource
from ..env import EnvironmentConfig, TradingEnv
from ..features import FeatureBuilder
from ..policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy
from .dqn_agent import DQNAgent
from .env_adapter import DQNPolicy, DiscreteTradingEnv
from .replay_buffer import ReplayBuffer
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation


@dataclass(slots=True)
class DQNConfig:
    policy_name: str | None = None
    total_steps: int = 20_000
    optimizer_updates: int | None = None
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 2_000
    train_every: int = 1
    target_update_interval: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    tau: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    double_dqn: bool = False
    dueling: bool = False
    network_type: str = "mlp"
    recurrent_hidden_size: int = 128
    recurrent_layers: int = 2
    recurrent_layer_sizes: tuple[int, ...] | None = None
    recurrent_dropout: float = 0.0
    head_dropout: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    validation_interval: int = 2_000
    selection_mode: str = "validation"
    validation_split_name: str = "val"
    early_stopping_patience_epochs: int | None = None
    epoch_steps: int | None = None
    reward_mode: str = "zhang"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seed: int = 7
    device: str = "auto"
    checkpoint_metric: str = "sharpe"


@dataclass(slots=True)
class TrainingArtifacts:
    artifact_dir: Path
    config_path: Path
    training_curve_path: Path
    summary_metrics_path: Path
    best_checkpoint_path: Path
    final_checkpoint_path: Path
    zhang_eval_dir: Path | None = None


def _coerce_loaded_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("date", "window_start_date", "window_end_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "window_ready" in frame.columns:
        normalized = frame["window_ready"].astype(str).str.lower()
        frame["window_ready"] = normalized.map({"true": True, "false": False}).fillna(False)
    return frame


def load_or_build_features(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    rebuild_data: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    features_path = output_path / "processed" / "features.csv"
    if features_path.exists() and not rebuild_data:
        features = pd.read_csv(features_path)
        return _coerce_loaded_feature_frame(features)

    pipeline = MarketDataPipeline(output_dir=output_path)
    artifacts = pipeline.build(
        source=PublicDailySource(),
        feature_builder=FeatureBuilder(),
        symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
    )
    return artifacts.features.copy()


class DQNTrainer:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        config: DQNConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.config = config or DQNConfig()
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
            action_mode="discrete",
            reward_mode=self.config.reward_mode,
            cost_rate_bp=self.config.cost_rate_bp,
            vol_target=self.config.vol_target,
        )
        self.rl_env = DiscreteTradingEnv(
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

        self.agent = DQNAgent(
            state_size=self.rl_env.state_size,
            action_size=self.rl_env.action_size,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            tau=self.config.tau,
            target_update_interval=self.config.target_update_interval,
            device=self.config.device,
            double_dqn=self.config.double_dqn,
            dueling=self.config.dueling,
            network_type=self.config.network_type,
            observation_window=self.rl_env.observation_window,
            feature_size=len(self.rl_env.feature_columns),
            recurrent_hidden_size=self.config.recurrent_hidden_size,
            recurrent_layers=self.config.recurrent_layers,
            recurrent_layer_sizes=self.config.recurrent_layer_sizes,
            recurrent_dropout=self.config.recurrent_dropout,
            head_dropout=self.config.head_dropout,
        )
        self.buffer = ReplayBuffer(
            capacity=self.config.replay_capacity,
            state_size=self.rl_env.state_size,
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
        index = int(self.rng.integers(len(candidates)))
        return candidates[index]

    def _epsilon(self, step: int) -> float:
        if self.config.epsilon_decay_steps <= 0:
            return float(self.config.epsilon_end)
        progress = min(step / self.config.epsilon_decay_steps, 1.0)
        return float(
            self.config.epsilon_start
            + progress * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def _write_frame_csv_atomic(self, path: Path, frame: pd.DataFrame) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _current_policy(self) -> DQNPolicy:
        return DQNPolicy(
            agent=self.agent,
            observation_window=self.rl_env.observation_window,
            feature_columns=self.rl_env.feature_columns,
            name=self._policy_name(),
        )

    def _policy_name(self) -> str:
        if self.config.policy_name:
            return self.config.policy_name
        pieces: list[str] = []
        if self.config.network_type.lower() == "lstm":
            pieces.append("lstm")
        if self.config.double_dqn:
            pieces.append("double")
        if self.config.dueling:
            pieces.append("dueling")
        pieces.append("dqn")
        return "_".join(pieces)

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
            "action_values": [-1.0, 0.0, 1.0],
        }
        self._write_json_atomic(config_path, payload)
        return config_path

    def _save_report_files(self, report) -> None:
        prefix = f"{report.policy_name}_{report.split}"
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_daily_returns.csv", report.daily_returns)
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_trades.csv", report.trade_log)
        self._write_frame_csv_atomic(self.report_dir / f"{prefix}_symbol_metrics.csv", report.symbol_metrics)

    def _default_epoch_steps(self) -> int:
        train_split_name = "train"
        start_date, end_date = self.splits[train_split_name]
        frame = self.feature_frame.loc[
            self.feature_frame["symbol"].isin(self.symbols_by_split.get(train_split_name, []))
            & (self.feature_frame["date"] >= pd.Timestamp(start_date))
            & (self.feature_frame["date"] <= pd.Timestamp(end_date))
            & self.feature_frame["window_ready"]
        ]
        return max(1, int(len(frame)))

    def train(self) -> TrainingArtifacts:
        config_path = self._persist_config()
        progress_rows: list[dict[str, object]] = []
        best_metric = float("-inf")
        has_validation = False
        update_count = 0
        patience = self.config.early_stopping_patience_epochs
        epochs_since_improvement = 0
        validation_interval = int(self.config.validation_interval) if self.config.validation_interval > 0 else 0
        if validation_interval <= 0 and self.config.selection_mode == "validation":
            validation_interval = int(self.config.epoch_steps or self._default_epoch_steps())
        current_epoch = 0

        state, _ = self.rl_env.reset(symbol=self._sample_symbol("train"), split="train")
        episode_return = 0.0
        episode_length = 0

        step = 0
        while True:
            step += 1
            epsilon = self._epsilon(step - 1)
            if self.rng.random() < epsilon:
                action_id = int(self.rng.integers(self.rl_env.action_size))
            else:
                action_id = self.agent.get_action(state)

            next_state, reward, terminated, truncated, _ = self.rl_env.step(action_id)
            done = terminated or truncated
            self.buffer.add((state, action_id, reward, next_state, done))
            state = next_state
            episode_return += reward
            episode_length += 1

            if (
                step >= self.config.warmup_steps
                and len(self.buffer) >= self.config.batch_size
                and step % self.config.train_every == 0
            ):
                update_info = self.agent.update(self.buffer.sample(self.config.batch_size), step=step)
                update_count += 1
                progress_rows.append(
                    {
                        "event": "update",
                        "step": step,
                        "epsilon": epsilon,
                        "loss": update_info["loss"],
                        "mean_q": update_info["mean_q"],
                        "mean_target": update_info["mean_target"],
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
                        "epsilon": epsilon,
                        "loss": np.nan,
                        "mean_q": np.nan,
                        "mean_target": np.nan,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "validation_metric": np.nan,
                    }
                )
                state, _ = self.rl_env.reset(symbol=self._sample_symbol("train"), split="train")
                episode_return = 0.0
                episode_length = 0

            if (
                self.config.selection_mode == "validation"
                and validation_interval > 0
                and step % validation_interval == 0
                and self.symbols_by_split.get(self.config.validation_split_name)
            ):
                current_epoch += 1
                report = self._evaluate_split(self._current_policy(), split=self.config.validation_split_name)
                if report is not None:
                    has_validation = True
                    metric = float(report.portfolio_metrics[self.config.checkpoint_metric])
                    progress_rows.append(
                        {
                            "event": "validation",
                            "step": step,
                            "epoch": current_epoch,
                            "epsilon": epsilon,
                            "loss": np.nan,
                            "mean_q": np.nan,
                            "mean_target": np.nan,
                            "episode_return": np.nan,
                            "episode_length": np.nan,
                            "validation_metric": metric,
                        }
                    )
                    if metric > best_metric:
                        best_metric = metric
                        self.agent.save(self.best_checkpoint_path)
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                    if patience is not None and epochs_since_improvement >= int(patience):
                        break

            if self.config.optimizer_updates is not None:
                if update_count >= self.config.optimizer_updates:
                    break
            elif step >= self.config.total_steps:
                break

        self.agent.save(self.final_checkpoint_path)
        if self.config.selection_mode == "final" or not has_validation:
            self.agent.save(self.best_checkpoint_path)
        self.agent.load(self.best_checkpoint_path)

        policies = [
            self._current_policy(),
            LongOnlyPolicy(),
            Sign12MPolicy(),
            MACDPolicy(),
        ]
        summary_rows: list[dict[str, object]] = []
        for split in self.splits:
            if not self.symbols_by_split.get(split):
                continue
            for policy in policies:
                report = self._evaluate_split(policy=policy, split=split)
                if report is None:
                    continue
                summary_rows.append(
                    {
                        "policy": report.policy_name,
                        "split": split,
                        **report.portfolio_metrics,
                    }
                )
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
                return_column="trade_return" if self.env_config.reward_mode == "zhang" else "raw_pnl",
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


def train_dqn(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: DQNConfig | None = None,
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
    trainer = DQNTrainer(
        feature_frame=features,
        output_dir=output_dir,
        config=config,
        splits=splits,
        symbols=symbols,
    )
    return trainer.train()
