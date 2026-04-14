"""Gym-style adapter around the trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OBSERVATION_WINDOW
from ..env import EnvironmentConfig, TradingEnv

ACTION_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def infer_observation_window(feature_frame: pd.DataFrame) -> int:
    if "window_size" not in feature_frame.columns:
        return OBSERVATION_WINDOW
    window_sizes = pd.to_numeric(feature_frame["window_size"], errors="coerce").dropna()
    if window_sizes.empty:
        return OBSERVATION_WINDOW
    return int(window_sizes.max())


def encode_observation(
    observation: dict[str, object],
    observation_window: int,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    num_features = len(feature_columns)
    padded_window = np.zeros((observation_window, num_features), dtype=np.float32)
    window = np.asarray(observation["window"], dtype=np.float32)
    if window.size:
        if window.ndim != 2 or window.shape[1] != num_features:
            raise ValueError("observation window does not match expected feature shape")
        valid_rows = min(observation_window, window.shape[0])
        padded_window[-valid_rows:] = window[-valid_rows:]

    position = np.array([float(observation["position"])], dtype=np.float32)
    state = np.concatenate([padded_window.reshape(-1), position])
    return state.astype(np.float32, copy=False)


class DiscreteTradingEnv:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        config: EnvironmentConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        observation_window: int | None = None,
    ) -> None:
        self.env = TradingEnv(feature_frame=feature_frame, config=config, splits=splits)
        self.feature_columns = self.env.config.feature_columns
        self.observation_window = observation_window or infer_observation_window(feature_frame)
        self.state_size = self.observation_window * len(self.feature_columns) + 1
        self.action_size = len(ACTION_VALUES)

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.state_size, dtype=np.float32)

    def reset(self, symbol: str, split: str) -> tuple[np.ndarray, dict[str, object]]:
        observation = self.env.reset(symbol=symbol, split=split)
        state = encode_observation(
            observation=observation,
            observation_window=self.observation_window,
            feature_columns=self.feature_columns,
        )
        info = {
            "symbol": symbol,
            "split": split,
            "date": pd.Timestamp(observation["date"]),
        }
        return state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_id = int(action)
        if action_id < 0 or action_id >= self.action_size:
            raise ValueError(f"action_id {action_id} is outside the valid range")

        target_position = float(ACTION_VALUES[action_id])
        next_observation, reward, done, info = self.env.step(target_position)
        next_state = self.zero_state()
        if not done and next_observation is not None:
            next_state = encode_observation(
                observation=next_observation,
                observation_window=self.observation_window,
                feature_columns=self.feature_columns,
            )

        step_info = dict(info)
        step_info["action_id"] = action_id
        step_info["target_position"] = target_position
        return next_state, float(reward), bool(done), False, step_info


class ContinuousTradingEnv:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        config: EnvironmentConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
        observation_window: int | None = None,
    ) -> None:
        continuous_config = config or EnvironmentConfig(action_mode="continuous")
        if continuous_config.action_mode != "continuous":
            continuous_config = EnvironmentConfig(
                action_mode="continuous",
                reward_mode=continuous_config.reward_mode,
                cost_rate_bp=continuous_config.cost_rate_bp,
                vol_target=continuous_config.vol_target,
                feature_columns=continuous_config.feature_columns,
            )
        self.env = TradingEnv(feature_frame=feature_frame, config=continuous_config, splits=splits)
        self.feature_columns = self.env.config.feature_columns
        self.observation_window = observation_window or infer_observation_window(feature_frame)
        self.state_size = self.observation_window * len(self.feature_columns) + 1
        self.action_size = 1

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.state_size, dtype=np.float32)

    def reset(self, symbol: str, split: str) -> tuple[np.ndarray, dict[str, object]]:
        observation = self.env.reset(symbol=symbol, split=split)
        state = encode_observation(
            observation=observation,
            observation_window=self.observation_window,
            feature_columns=self.feature_columns,
        )
        info = {
            "symbol": symbol,
            "split": split,
            "date": pd.Timestamp(observation["date"]),
        }
        return state, info

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        target_position = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        next_observation, reward, done, info = self.env.step(target_position)
        next_state = self.zero_state()
        if not done and next_observation is not None:
            next_state = encode_observation(
                observation=next_observation,
                observation_window=self.observation_window,
                feature_columns=self.feature_columns,
            )
        step_info = dict(info)
        step_info["target_position"] = float(np.clip(target_position, -1.0, 1.0))
        return next_state, float(reward), bool(done), False, step_info


class DQNPolicy:
    def __init__(
        self,
        agent,
        observation_window: int,
        feature_columns: tuple[str, ...],
        name: str = "dqn",
    ) -> None:
        self.agent = agent
        self.observation_window = observation_window
        self.feature_columns = feature_columns
        self.name = name

    def act(self, observation: dict[str, object]) -> float:
        state = encode_observation(
            observation=observation,
            observation_window=self.observation_window,
            feature_columns=self.feature_columns,
        )
        action_id = self.agent.get_action(state)
        return float(ACTION_VALUES[action_id])


class DiscreteActorPolicy:
    def __init__(
        self,
        agent,
        observation_window: int,
        feature_columns: tuple[str, ...],
        name: str = "ppo",
    ) -> None:
        self.agent = agent
        self.observation_window = observation_window
        self.feature_columns = feature_columns
        self.name = name

    def act(self, observation: dict[str, object]) -> float:
        state = encode_observation(
            observation=observation,
            observation_window=self.observation_window,
            feature_columns=self.feature_columns,
        )
        action_id = self.agent.get_greedy_action(state)
        return float(ACTION_VALUES[action_id])


class ContinuousActorPolicy:
    def __init__(
        self,
        agent,
        observation_window: int,
        feature_columns: tuple[str, ...],
        name: str = "continuous_rl",
    ) -> None:
        self.agent = agent
        self.observation_window = observation_window
        self.feature_columns = feature_columns
        self.name = name

    def act(self, observation: dict[str, object]) -> float:
        state = encode_observation(
            observation=observation,
            observation_window=self.observation_window,
            feature_columns=self.feature_columns,
        )
        if hasattr(self.agent, "get_action"):
            try:
                action = self.agent.get_action(state, deterministic=True)
            except TypeError:
                action = self.agent.get_action(state)
            if isinstance(action, tuple):
                action = action[0]
            return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        raise TypeError("agent does not expose get_action")
