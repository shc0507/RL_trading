"""Production Zhang-style continuous A2C trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from ..backtest import Backtester
from ..config import DEFAULT_COST_RATE_BP, DEFAULT_OUTPUT_DIR, DEFAULT_VOL_TARGET
from ..env import EnvironmentConfig, TradingEnv
from ..policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy
from .env_adapter import ContinuousActorPolicy, ContinuousTradingEnv
from .policy_gradient import compute_gae
from .sequence_encoder import StackedLSTMStateEncoder
from .torch_utils import as_float_tensor, build_mlp, resolve_device
from .trainer import TrainingArtifacts
from .zhang_eval import ZhangEvalConfig, run_zhang_evaluation


@dataclass(slots=True)
class ProductionA2CConfig:
    policy_name: str | None = None
    total_steps: int = 250_000
    batch_size: int = 128
    rollout_steps: int | None = None
    gamma: float = 0.3
    gae_lambda: float = 0.95
    hidden_sizes: tuple[int, ...] = (32,)
    activation: str = "leaky_relu"
    recurrent_dropout: float = 0.1
    head_dropout: float = 0.1
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float | None = 0.5
    reward_mode: str = "zhang"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seed: int = 7
    device: str = "auto"
    recurrent_layer_sizes: tuple[int, ...] = (64, 32)
    validation_split_name: str = "cv"
    selection_metric: str = "sharpe"
    early_stopping_patience_epochs: int | None = 20
    epoch_steps: int | None = None
    search_grid: tuple[dict[str, object], ...] | None = None


@dataclass(slots=True)
class ContinuousOnPolicyBatch:
    states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class ContinuousGaussianActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        observation_window: int,
        feature_size: int,
        recurrent_layer_sizes: tuple[int, ...] = (64, 32),
        hidden_sizes: tuple[int, ...] = (32,),
        activation: str = "leaky_relu",
        log_std_init: float = -0.5,
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
        self.mean_head = build_mlp(
            input_size=self.encoder.output_size,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=head_dropout,
        )
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init), dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(states)
        mean = self.mean_head(latent)
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 2.0)
        return mean, log_std


class ContinuousValueCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
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
        self.value_head = build_mlp(
            input_size=self.encoder.output_size,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=head_dropout,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(states)
        return self.value_head(latent).squeeze(-1)


class ContinuousA2CAgent:
    def __init__(
        self,
        state_size: int,
        observation_window: int,
        feature_size: int,
        recurrent_layer_sizes: tuple[int, ...] = (64, 32),
        hidden_sizes: tuple[int, ...] = (32,),
        activation: str = "leaky_relu",
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float | None = 0.5,
        device: str = "auto",
        recurrent_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        self.device = resolve_device(device)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = max_grad_norm
        self.actor = ContinuousGaussianActor(
            state_size=state_size,
            observation_window=observation_window,
            feature_size=feature_size,
            recurrent_layer_sizes=recurrent_layer_sizes,
            hidden_sizes=hidden_sizes,
            activation=activation,
            recurrent_dropout=recurrent_dropout,
            head_dropout=head_dropout,
        ).to(self.device)
        self.critic = ContinuousValueCritic(
            state_size=state_size,
            observation_window=observation_window,
            feature_size=feature_size,
            recurrent_layer_sizes=recurrent_layer_sizes,
            hidden_sizes=hidden_sizes,
            activation=activation,
            recurrent_dropout=recurrent_dropout,
            head_dropout=head_dropout,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _distribution(self, states: torch.Tensor) -> tuple[distributions.Normal, torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(states)
        normal = distributions.Normal(mean, log_std.exp())
        return normal, mean, log_std

    def _sample_actions(
        self,
        states: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, mean, _ = self._distribution(states)
        raw_action = mean if deterministic else normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = normal.entropy()
        return action, log_prob.sum(dim=-1), entropy.sum(dim=-1)

    @torch.no_grad()
    def sample_actions(
        self,
        states: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tensor = as_float_tensor(states, self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action, log_prob, _ = self._sample_actions(state_tensor, deterministic=deterministic)
        value = self.critic(state_tensor)
        return (
            action.squeeze(-1).cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    @torch.no_grad()
    def get_action(self, state, deterministic: bool = False) -> tuple[float, float, float]:
        actions, log_probs, values = self.sample_actions(state, deterministic=deterministic)
        return float(actions.reshape(-1)[0]), float(log_probs.reshape(-1)[0]), float(values.reshape(-1)[0])

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, _, _ = self._distribution(states)
        clipped = actions.clamp(-0.999999, 0.999999)
        raw_action = torch.atanh(clipped)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - clipped.pow(2) + 1e-6)
        entropy = normal.entropy().sum(dim=-1)
        value = self.critic(states)
        return log_prob.sum(dim=-1), entropy, value

    def update(self, batch: ContinuousOnPolicyBatch) -> dict[str, float]:
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device).float()
        returns = batch.returns.to(self.device).float()
        advantages = batch.advantages.to(self.device).float()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        log_prob, entropy, values = self._evaluate_actions(states, actions)
        actor_loss = -(log_prob * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

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
            "entropy": float(entropy.mean().item()),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, target)

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


class ProductionA2CTrainer:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        config: ProductionA2CConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.config = config or ProductionA2CConfig()
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
            action_mode="continuous",
            reward_mode=self.config.reward_mode,
            cost_rate_bp=self.config.cost_rate_bp,
            vol_target=self.config.vol_target,
        )
        self.eval_env = ContinuousTradingEnv(
            feature_frame=self.feature_frame,
            config=self.env_config,
            splits=self.splits,
        )
        self.backtester = Backtester(
            feature_frame=self.feature_frame,
            env_config=self.env_config,
            splits=self.splits,
        )
        self.symbols_by_split = {split_name: self._resolve_valid_symbols(split_name) for split_name in self.splits}
        if not self.symbols_by_split.get("train"):
            raise ValueError("no valid symbols are available for the train split")

        self.agent = ContinuousA2CAgent(
            state_size=self.eval_env.state_size,
            observation_window=self.eval_env.observation_window,
            feature_size=len(self.eval_env.feature_columns),
            recurrent_layer_sizes=self.config.recurrent_layer_sizes,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            recurrent_dropout=self.config.recurrent_dropout,
            head_dropout=self.config.head_dropout,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            entropy_coef=self.config.entropy_coef,
            value_coef=self.config.value_coef,
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

    def _write_frame_csv_atomic(self, path: Path, frame: pd.DataFrame) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _policy_name(self) -> str:
        return self.config.policy_name or "a2c"

    def _current_policy(self) -> ContinuousActorPolicy:
        return ContinuousActorPolicy(
            agent=self.agent,
            observation_window=self.eval_env.observation_window,
            feature_columns=self.eval_env.feature_columns,
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
            "state_size": self.eval_env.state_size,
            "action_range": [-1.0, 1.0],
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

    @torch.no_grad()
    def _next_value(self, state: np.ndarray, done: bool) -> torch.Tensor:
        if done:
            return torch.zeros((), dtype=torch.float32, device=self.agent.device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
        return self.agent.critic(state_tensor).reshape(())

    def _rollout_batch(
        self,
        trajectories: list[dict[str, list[object]]],
        current_states: list[np.ndarray],
    ) -> ContinuousOnPolicyBatch:
        device = self.agent.device
        batch_states: list[np.ndarray] = []
        batch_actions: list[float] = []
        batch_returns: list[torch.Tensor] = []
        batch_advantages: list[torch.Tensor] = []
        for trajectory, current_state in zip(trajectories, current_states, strict=True):
            if not trajectory["states"]:
                continue
            rewards = torch.as_tensor(trajectory["rewards"], dtype=torch.float32, device=device)
            dones = torch.as_tensor(trajectory["dones"], dtype=torch.float32, device=device)
            values = torch.as_tensor(trajectory["values"], dtype=torch.float32, device=device)
            next_value = self._next_value(current_state, bool(trajectory["dones"][-1]))
            advantages, returns = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                next_value=next_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            batch_states.extend(trajectory["states"])
            batch_actions.extend(trajectory["actions"])
            batch_returns.append(returns.detach())
            batch_advantages.append(advantages.detach())

        state_tensor = torch.as_tensor(np.asarray(batch_states, dtype=np.float32), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(batch_actions, dtype=torch.float32, device=device).unsqueeze(-1)
        return_tensor = torch.cat(batch_returns, dim=0)
        advantage_tensor = torch.cat(batch_advantages, dim=0)
        return ContinuousOnPolicyBatch(
            states=state_tensor,
            actions=action_tensor,
            returns=return_tensor,
            advantages=advantage_tensor,
        )

    def train(self) -> TrainingArtifacts:
        config_path = self._persist_config()
        progress_rows: list[dict[str, object]] = []
        best_metric = float("-inf")
        epochs_since_improvement = 0
        current_epoch = 0
        epoch_steps = int(self.config.epoch_steps or self._default_epoch_steps())
        patience = self.config.early_stopping_patience_epochs

        train_symbols = self.symbols_by_split["train"]
        envs = [
            ContinuousTradingEnv(feature_frame=self.feature_frame, config=self.env_config, splits=self.splits)
            for _ in train_symbols
        ]
        current_states: list[np.ndarray] = []
        episode_returns = [0.0 for _ in train_symbols]
        episode_lengths = [0 for _ in train_symbols]
        for env, symbol in zip(envs, train_symbols, strict=True):
            state, _ = env.reset(symbol=symbol, split="train")
            current_states.append(state)

        rollout_steps = self.config.rollout_steps or max(1, math.ceil(self.config.batch_size / len(envs)))
        total_steps_done = 0
        while total_steps_done < self.config.total_steps:
            trajectories = [
                {"states": [], "actions": [], "rewards": [], "dones": [], "values": []}
                for _ in envs
            ]
            for _ in range(rollout_steps):
                state_batch = np.stack(current_states, axis=0)
                actions, _, values = self.agent.sample_actions(state_batch, deterministic=False)
                for index, env in enumerate(envs):
                    action = float(actions[index])
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    trajectories[index]["states"].append(current_states[index].copy())
                    trajectories[index]["actions"].append(action)
                    trajectories[index]["rewards"].append(float(reward))
                    trajectories[index]["dones"].append(done)
                    trajectories[index]["values"].append(float(values[index]))
                    episode_returns[index] += float(reward)
                    episode_lengths[index] += 1
                    total_steps_done += 1
                    if done:
                        progress_rows.append(
                            {
                                "event": "episode",
                                "step": total_steps_done,
                                "loss": np.nan,
                                "actor_loss": np.nan,
                                "value_loss": np.nan,
                                "entropy": np.nan,
                                "episode_return": episode_returns[index],
                                "episode_length": episode_lengths[index],
                            }
                        )
                        reset_state, _ = env.reset(symbol=train_symbols[index], split="train")
                        current_states[index] = reset_state
                        episode_returns[index] = 0.0
                        episode_lengths[index] = 0
                    else:
                        current_states[index] = next_state
                    if total_steps_done >= self.config.total_steps:
                        break
                if total_steps_done >= self.config.total_steps:
                    break

            batch = self._rollout_batch(trajectories=trajectories, current_states=current_states)
            update_info = self.agent.update(batch)
            progress_rows.append(
                {
                    "event": "update",
                    "step": total_steps_done,
                    "epoch": current_epoch,
                    "loss": update_info.get("loss", np.nan),
                    "actor_loss": update_info.get("actor_loss", np.nan),
                    "value_loss": update_info.get("value_loss", np.nan),
                    "entropy": update_info.get("entropy", np.nan),
                    "episode_return": np.nan,
                    "episode_length": np.nan,
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
                                "actor_loss": np.nan,
                                "value_loss": np.nan,
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
                        if patience is not None and epochs_since_improvement >= int(patience):
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
            config=ZhangEvalConfig(
                target_vol=self.config.vol_target,
                split=None,
                return_column="trade_return",
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
