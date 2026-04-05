"""Gymnasium wrapper for TradingEnv so SB3 can train on the existing trading logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .backtest import Backtester, EvalReport
from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_SPLITS, OBSERVATION_WINDOW
from .env import EnvironmentConfig, TradingEnv
from .policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy


DISCRETE_ACTIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def encode_observation(
    observation: dict[str, object],
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    include_position: bool = True,
) -> np.ndarray:
    window = np.asarray(observation["window"], dtype=np.float32)
    feature_count = len(feature_columns)
    if window.ndim != 2 or window.shape[1] != feature_count:
        raise ValueError(
            f"expected observation window with shape (*, {feature_count}), got {window.shape}"
        )
    if window.shape[0] < OBSERVATION_WINDOW:
        pad = np.zeros((OBSERVATION_WINDOW - window.shape[0], feature_count), dtype=np.float32)
        window = np.vstack([pad, window])
    elif window.shape[0] > OBSERVATION_WINDOW:
        window = window[-OBSERVATION_WINDOW:]
    flattened = window.reshape(-1)
    if include_position:
        flattened = np.concatenate([flattened, np.array([float(observation["position"])], dtype=np.float32)])
    return flattened.astype(np.float32, copy=False)


@dataclass(slots=True)
class SB3PolicyAdapter:
    model: object
    action_mode: Literal["continuous", "discrete"]
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    include_position: bool = True
    name: str = "sb3_model"

    def act(self, observation: dict[str, object]) -> float:
        obs_vec = encode_observation(
            observation,
            feature_columns=self.feature_columns,
            include_position=self.include_position,
        )
        action, _ = self.model.predict(obs_vec, deterministic=True)
        if self.action_mode == "discrete":
            return float(DISCRETE_ACTIONS[int(action)])
        return float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))


class SB3TradingEnv(gym.Env):
    """Thin Gymnasium adapter over TradingEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        split: str = "train",
        symbols: list[str] | None = None,
        action_mode: Literal["continuous", "discrete"] = "discrete",
        reward_mode: Literal["raw", "zhang"] = "zhang",
        cost_rate_bp: float = 20.0,
        splits: dict[str, tuple[str, str]] | None = None,
        feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
        include_position: bool = True,
        shuffle_symbols: bool = True,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.split = split
        self.symbols = symbols or sorted(self.feature_frame["symbol"].unique().tolist())
        self.action_mode = action_mode
        self.reward_mode = reward_mode
        self.cost_rate_bp = cost_rate_bp
        self.splits = splits or DEFAULT_SPLITS
        self.feature_columns = feature_columns
        self.include_position = include_position
        self.shuffle_symbols = shuffle_symbols
        self.rng = np.random.default_rng(seed)
        self.current_symbol: str | None = None

        self.core_env = TradingEnv(
            feature_frame=self.feature_frame,
            config=EnvironmentConfig(
                action_mode=action_mode,
                reward_mode=reward_mode,
                cost_rate_bp=cost_rate_bp,
                feature_columns=feature_columns,
            ),
            splits=self.splits,
        )

        obs_size = OBSERVATION_WINDOW * len(self.feature_columns) + (1 if include_position else 0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        chosen_symbol = None
        if options is not None and "symbol" in options:
            chosen_symbol = str(options["symbol"])
        elif self.shuffle_symbols:
            chosen_symbol = str(self.rng.choice(self.symbols))
        else:
            chosen_symbol = self.symbols[0]

        self.current_symbol = chosen_symbol
        observation = self.core_env.reset(symbol=chosen_symbol, split=self.split)
        encoded = encode_observation(
            observation,
            feature_columns=self.feature_columns,
            include_position=self.include_position,
        )
        info = {"symbol": chosen_symbol, "split": self.split}
        return encoded, info

    def step(self, action):
        if self.action_mode == "discrete":
            target_position = float(DISCRETE_ACTIONS[int(action)])
        else:
            target_position = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))

        next_observation, reward, done, info = self.core_env.step(target_position)
        if done or next_observation is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = dict(info)
            info["terminal_observation"] = obs.copy()
        else:
            obs = encode_observation(
                next_observation,
                feature_columns=self.feature_columns,
                include_position=self.include_position,
            )
        return obs, float(reward), bool(done), False, info


def compare_sb3_policy(
    feature_frame: pd.DataFrame,
    model: object,
    name: str,
    split: str,
    action_mode: Literal["continuous", "discrete"],
    reward_mode: Literal["raw", "zhang"] = "raw",
    cost_rate_bp: float = 20.0,
    symbols: list[str] | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    include_position: bool = True,
) -> tuple[pd.DataFrame, dict[str, EvalReport]]:
    env_config = EnvironmentConfig(
        action_mode=action_mode,
        reward_mode=reward_mode,
        cost_rate_bp=cost_rate_bp,
        feature_columns=feature_columns,
    )
    backtester = Backtester(feature_frame=feature_frame, env_config=env_config, splits=splits)
    learned_policy = SB3PolicyAdapter(
        model=model,
        action_mode=action_mode,
        feature_columns=feature_columns,
        include_position=include_position,
        name=name,
    )
    policies = [learned_policy, LongOnlyPolicy(), Sign12MPolicy(), MACDPolicy()]
    comparison_rows: list[dict[str, float | str]] = []
    reports: dict[str, EvalReport] = {}

    for policy in policies:
        report = backtester.run(policy=policy, split=split, symbols=symbols)
        reports[policy.name] = report
        comparison_rows.append({"policy": policy.name, "split": split, **report.portfolio_metrics})

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(["annualized_return", "sharpe"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison, reports
