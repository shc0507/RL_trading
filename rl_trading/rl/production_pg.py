"""Production Zhang-style recurrent policy-gradient trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import distributions, nn, optim

from ..backtest import Backtester
from ..config import DEFAULT_COST_RATE_BP, DEFAULT_OUTPUT_DIR, DEFAULT_VOL_TARGET
from ..env import EnvironmentConfig, TradingEnv
from ..policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy
from .env_adapter import DiscreteActorPolicy, DiscreteTradingEnv
from .policy_gradient import discounted_reward_to_go
from .sequence_encoder import StackedLSTMStateEncoder
from .torch_utils import as_float_tensor, build_mlp, resolve_device
from .trainer import TrainingArtifacts
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation


@dataclass(slots=True)
class ProductionPGConfig:
    policy_name: str | None = None
    total_steps: int = 250_000
    gamma: float = 0.3
    hidden_sizes: tuple[int, ...] = (32,)
    activation: str = "leaky_relu"
    recurrent_layer_sizes: tuple[int, ...] = (64, 32)
    recurrent_dropout: float = 0.1
    head_dropout: float = 0.1
    actor_lr: float = 1e-4
    entropy_coef: float = 0.01
    max_grad_norm: float | None = 0.5
    reward_mode: str = "zhang"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seed: int = 11
    device: str = "auto"
    validation_split_name: str = "cv"
    selection_metric: str = "sharpe"
    early_stopping_patience_epochs: int | None = 20
    epoch_steps: int | None = None
    search_grid: tuple[dict[str, object], ...] | None = None


class RecurrentCategoricalActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        observation_window: int,
        feature_size: int,
        recurrent_layer_sizes: tuple[int, ...] = (64, 32),
        hidden_sizes: tuple[int, ...] = (32,),
        activation: str = "leaky_relu",
        recurrent_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = StackedLSTMStateEncoder(
            state_size=state_size,
            observation_window=observation_window,
            feature_size=feature_size,
            recurrent_layer_sizes=recurrent_layer_sizes,
            recurrent_dropout=recurrent_dropout,
        )
        self.logits_net = build_mlp(
            input_size=self.encoder.output_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=head_dropout,
        )

    def forward(self, states: torch.Tensor) -> distributions.Categorical:
        latent = self.encoder(states)
        return distributions.Categorical(logits=self.logits_net(latent))


class ProductionPGAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        observation_window: int,
        feature_size: int,
        recurrent_layer_sizes: tuple[int, ...] = (64, 32),
        hidden_sizes: tuple[int, ...] = (32,),
        activation: str = "leaky_relu",
        recurrent_dropout: float = 0.0,
        head_dropout: float = 0.0,
        actor_lr: float = 1e-4,
        entropy_coef: float = 0.01,
        max_grad_norm: float | None = 0.5,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = max_grad_norm
        self.actor = RecurrentCategoricalActor(
            state_size=state_size,
            action_size=action_size,
            observation_window=observation_window,
            feature_size=feature_size,
            recurrent_layer_sizes=recurrent_layer_sizes,
            hidden_sizes=hidden_sizes,
            activation=activation,
            recurrent_dropout=recurrent_dropout,
            head_dropout=head_dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    @torch.no_grad()
    def get_action(self, state, deterministic: bool = False) -> tuple[int, float]:
        state_tensor = as_float_tensor(state, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        distribution = self.actor(state_tensor)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item())

    @torch.no_grad()
    def get_greedy_action(self, state) -> int:
        action, _ = self.get_action(state, deterministic=True)
        return action

    def update(self, states: list[np.ndarray], actions: list[int], rewards: list[float], gamma: float) -> dict[str, float]:
        returns = discounted_reward_to_go(np.asarray(rewards, dtype=np.float32), gamma=float(gamma))
        state_tensor = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if return_tensor.numel() > 1:
            return_tensor = (return_tensor - return_tensor.mean()) / (return_tensor.std(unbiased=False) + 1e-8)
        distribution = self.actor(state_tensor)
        log_probs = distribution.log_prob(action_tensor)
        entropy = distribution.entropy().mean()
        loss = -(log_probs * return_tensor).mean() - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "entropy": float(entropy.item()),
            "mean_return": float(torch.as_tensor(returns).mean().item()) if len(returns) else 0.0,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict()}, target)

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state["actor"])


class ProductionPGTrainer:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        config: ProductionPGConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.config = config or ProductionPGConfig()
        self.splits = splits or {}
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
        self.rl_env = DiscreteTradingEnv(feature_frame=self.feature_frame, config=self.env_config, splits=self.splits)
        self.backtester = Backtester(feature_frame=self.feature_frame, env_config=self.env_config, splits=self.splits)
        self.symbols_by_split = {split_name: self._resolve_valid_symbols(split_name) for split_name in self.splits}
        if not self.symbols_by_split.get("train"):
            raise ValueError("no valid symbols are available for the train split")

        self.agent = ProductionPGAgent(
            state_size=self.rl_env.state_size,
            action_size=self.rl_env.action_size,
            observation_window=self.rl_env.observation_window,
            feature_size=len(self.rl_env.feature_columns),
            recurrent_layer_sizes=self.config.recurrent_layer_sizes,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            recurrent_dropout=self.config.recurrent_dropout,
            head_dropout=self.config.head_dropout,
            actor_lr=self.config.actor_lr,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
            device=self.config.device,
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
        validation_env = TradingEnv(feature_frame=self.feature_frame, config=self.env_config, splits=self.splits)
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
        return self.config.policy_name or "pg"

    def _current_policy(self) -> DiscreteActorPolicy:
        return DiscreteActorPolicy(
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
        start_date, end_date = self.splits["train"]
        frame = self.feature_frame.loc[
            self.feature_frame["symbol"].isin(self.symbols_by_split.get("train", []))
            & (self.feature_frame["date"] >= pd.Timestamp(start_date))
            & (self.feature_frame["date"] <= pd.Timestamp(end_date))
            & self.feature_frame["window_ready"]
        ]
        return max(1, int(len(frame)))

    def train(self) -> TrainingArtifacts:
        config_path = self._persist_config()
        progress_rows: list[dict[str, object]] = []
        best_metric = float("-inf")
        epochs_since_improvement = 0
        current_epoch = 0
        epoch_steps = int(self.config.epoch_steps or self._default_epoch_steps())
        total_steps_done = 0

        while total_steps_done < self.config.total_steps:
            state, _ = self.rl_env.reset(symbol=self._sample_symbol("train"), split="train")
            episode_states: list[np.ndarray] = []
            episode_actions: list[int] = []
            episode_rewards: list[float] = []
            done = False
            while not done and total_steps_done < self.config.total_steps:
                action_id, _ = self.agent.get_action(state, deterministic=False)
                next_state, reward, terminated, truncated, _ = self.rl_env.step(action_id)
                episode_states.append(state.copy())
                episode_actions.append(action_id)
                episode_rewards.append(float(reward))
                state = next_state
                done = bool(terminated or truncated)
                total_steps_done += 1

            update_info = self.agent.update(
                states=episode_states,
                actions=episode_actions,
                rewards=episode_rewards,
                gamma=self.config.gamma,
            )
            progress_rows.append(
                {
                    "event": "update",
                    "step": total_steps_done,
                    "epoch": current_epoch,
                    "loss": update_info["loss"],
                    "entropy": update_info["entropy"],
                    "episode_return": float(sum(episode_rewards)),
                    "episode_length": len(episode_rewards),
                    "validation_metric": np.nan,
                }
            )

            while total_steps_done >= (current_epoch + 1) * epoch_steps:
                current_epoch += 1
                if self.symbols_by_split.get(self.config.validation_split_name):
                    report = self._evaluate_split(policy=self._current_policy(), split=self.config.validation_split_name)
                    if report is not None:
                        metric = float(report.portfolio_metrics[self.config.selection_metric])
                        progress_rows.append(
                            {
                                "event": "validation",
                                "step": total_steps_done,
                                "epoch": current_epoch,
                                "loss": np.nan,
                                "entropy": np.nan,
                                "episode_return": np.nan,
                                "episode_length": np.nan,
                                "validation_metric": metric,
                            }
                        )
                        if metric > best_metric:
                            best_metric = metric
                            epochs_since_improvement = 0
                            self.agent.save(self.best_checkpoint_path)
                        else:
                            epochs_since_improvement += 1
                        if (
                            self.config.early_stopping_patience_epochs is not None
                            and epochs_since_improvement >= int(self.config.early_stopping_patience_epochs)
                        ):
                            total_steps_done = self.config.total_steps
                            break

        self.agent.save(self.final_checkpoint_path)
        if best_metric == float("-inf"):
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
            config=ZhangEvalConfig(target_vol=self.config.vol_target, split=None, return_column="trade_return"),
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
