"""Single-symbol trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import DEFAULT_COST_RATE_BP, DEFAULT_FEATURE_COLUMNS, DEFAULT_SPLITS, DEFAULT_VOL_TARGET


@dataclass(slots=True)
class EnvironmentConfig:
    action_mode: Literal["continuous", "discrete"] = "continuous"
    reward_mode: Literal["raw", "zhang"] = "zhang"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS


class TradingEnv:
    """Close-to-close trading environment."""

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        config: EnvironmentConfig | None = None,
        splits: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.feature_frame = feature_frame.copy()
        self.feature_frame["date"] = pd.to_datetime(self.feature_frame["date"])
        self.config = config or EnvironmentConfig()
        self.splits = splits or DEFAULT_SPLITS
        self.episode_frame: pd.DataFrame | None = None
        self.symbol: str | None = None
        self.split: str | None = None
        self.pointer = 0
        self.position = 0.0
        self.current_vol: float | None = None

    def reset(self, symbol: str, split: str) -> dict[str, object]:
        if split not in self.splits:
            raise KeyError(f"unknown split {split}")
        start_date, end_date = self.splits[split]
        frame = self.feature_frame.loc[self.feature_frame["symbol"] == symbol].copy()
        if frame.empty:
            raise ValueError(f"no feature rows found for symbol {symbol}")
        mask = (
            (frame["date"] >= pd.Timestamp(start_date))
            & (frame["date"] <= pd.Timestamp(end_date))
            & frame["window_ready"]
        )
        episode = frame.loc[mask].reset_index(drop=True)
        if len(episode) < 2:
            raise ValueError(f"not enough rows for symbol {symbol} in split {split}")

        self.episode_frame = episode
        self.symbol = symbol
        self.split = split
        self.pointer = 0
        self.position = 0.0
        self.current_vol = None
        return self._observation()

    def step(self, target_position: float) -> tuple[dict[str, object] | None, float, bool, dict[str, object]]:
        if self.episode_frame is None:
            raise RuntimeError("reset must be called before step")
        if self.pointer >= len(self.episode_frame) - 1:
            raise RuntimeError("environment is already done")

        action = self._normalize_action(target_position)
        current = self.episode_frame.iloc[self.pointer]
        next_row = self.episode_frame.iloc[self.pointer + 1]
        cost_rate = self.config.cost_rate_bp / 10_000.0
        price_now = float(current["adj_close"])
        price_next = float(next_row["adj_close"])
        pct_change = (price_next / price_now) - 1.0

        turnover = abs(action - self.position)
        raw_pnl = action * (price_next - price_now) - (cost_rate * price_now * turnover)
        raw_return = action * pct_change - (cost_rate * turnover)

        current_vol = max(float(current["ewm_vol_60"]), 1e-8)
        previous_scaled_position = 0.0 if self.current_vol is None else self.position * (self.config.vol_target / self.current_vol)
        scaled_action = action * (self.config.vol_target / current_vol)
        scaled_turnover = abs(scaled_action - previous_scaled_position)
        zhang_reward = scaled_action * (price_next - price_now) - (cost_rate * price_now * scaled_turnover)
        zhang_return = scaled_action * pct_change - (cost_rate * scaled_turnover)

        self.position = action
        self.current_vol = current_vol
        self.pointer += 1
        done = self.pointer >= len(self.episode_frame) - 1
        reward = raw_pnl if self.config.reward_mode == "raw" else zhang_reward
        next_observation = None if done else self._observation()

        info = {
            "date": pd.Timestamp(current["date"]),
            "next_date": pd.Timestamp(next_row["date"]),
            "symbol": self.symbol,
            "split": self.split,
            "action": action,
            "turnover": turnover,
            "scaled_turnover": scaled_turnover,
            "raw_pnl": raw_pnl,
            "raw_return": raw_return,
            "raw_cost": cost_rate * price_now * turnover,
            "raw_cost_return": cost_rate * turnover,
            "zhang_reward": zhang_reward,
            "zhang_return": zhang_return,
            "zhang_cost": cost_rate * price_now * scaled_turnover,
            "zhang_cost_return": cost_rate * scaled_turnover,
            "price_now": price_now,
            "price_next": price_next,
            "pct_change": pct_change,
        }
        return next_observation, reward, done, info

    def _observation(self) -> dict[str, object]:
        if self.episode_frame is None:
            raise RuntimeError("episode not initialized")
        row = self.episode_frame.iloc[self.pointer]
        position = float(self.position)
        window_end = self.pointer + 1
        window_start = max(0, window_end - int(row["window_size"]))
        window = self.episode_frame.iloc[window_start:window_end]
        feature_window = window.loc[:, self.config.feature_columns].to_numpy(dtype=float)
        return {
            "symbol": self.symbol,
            "split": self.split,
            "date": pd.Timestamp(row["date"]),
            "position": position,
            "window": feature_window,
            "features": {column: float(row[column]) for column in self.config.feature_columns},
            "row": row.to_dict(),
        }

    def _normalize_action(self, target_position: float) -> float:
        action = float(np.clip(target_position, -1.0, 1.0))
        if self.config.action_mode == "discrete":
            discrete_choices = np.array([-1.0, 0.0, 1.0])
            action = float(discrete_choices[np.abs(discrete_choices - action).argmin()])
        return action
