"""Single-asset trading environment per Zhang et al. (2019)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_trading.config import (
    DEFAULT_COST_RATE_BP,
    DEFAULT_VOL_TARGET,
    OBSERVATION_WINDOW,
    TEST_COST_RATE_BP,
    TRAIN_END,
    TRAIN_START,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
)
from rl_trading.features import FEATURE_COLS

_SPLITS: dict[str, tuple[str, str]] = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}

_ACTION_TO_POS = {0: -1.0, 1: 0.0, 2: 1.0}

# 10 market features (Zhang 2019 p.4); position is consumed only via
# the reward (Eq. 4), not exposed to the agent.
STATE_DIM = len(FEATURE_COLS)


@dataclass
class EnvConfig:
    action_mode: str = "discrete"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    test_cost_rate_bp: float = TEST_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seq_len: int = OBSERVATION_WINDOW
    state_normalize: bool = False


class TradingEnv:
    """Single-asset trading environment with Zhang et al. reward."""

    def __init__(self, feature_frame: pd.DataFrame, cfg: EnvConfig | None = None):
        self.full_frame = feature_frame
        self.cfg = cfg or EnvConfig()
        self._bp = self.cfg.cost_rate_bp / 10_000

        self._data: pd.DataFrame | None = None
        self._features: np.ndarray | None = None
        self._prices: np.ndarray | None = None
        self._ewm_vol: np.ndarray | None = None
        self._dates: np.ndarray | None = None
        self._t: int = 0
        self._position: float = 0.0
        self._prev_vol_scale: float = 0.0

        self.history: dict[str, list] = {}

    def reset(
        self, symbol: str, split: str = "train",
        *, start: str | None = None, end: str | None = None,
    ) -> np.ndarray:
        """Start a new episode.

        Returns shape (seq_len, n_features) when ``cfg.seq_len > 1``,
        else (n_features,). ``self._bp`` is set from ``cost_rate_bp`` for
        train/val and ``test_cost_rate_bp`` otherwise.
        """
        if start is not None and end is not None:
            date_start, date_end = start, end
            active_bp_rate = self.cfg.test_cost_rate_bp
        else:
            date_start, date_end = _SPLITS[split]
            active_bp_rate = (
                self.cfg.test_cost_rate_bp if split == "test"
                else self.cfg.cost_rate_bp
            )
        self._bp = active_bp_rate / 10_000
        mask = (
            (self.full_frame["symbol"] == symbol)
            & (self.full_frame["window_ready"])
            & (self.full_frame["date"] >= date_start)
            & (self.full_frame["date"] <= date_end)
        )
        self._data = self.full_frame.loc[mask].sort_values("date").reset_index(drop=True)
        if len(self._data) == 0:
            raise ValueError(f"No data for {symbol} in split={split}")

        self._features = self._data[FEATURE_COLS].to_numpy(dtype=np.float32)
        self._prices = self._data["close"].to_numpy(dtype=np.float64)
        self._ewm_vol = self._data["ewm_vol"].to_numpy(dtype=np.float64)
        self._dates = self._data["date"].to_numpy()

        self._t = max(self.cfg.seq_len - 1, 0)
        self._position = 0.0
        self._prev_vol_scale = 0.0

        # Per-contract reward normalization (Zhang p.5 μ knob): with μ=1
        # the equal-weight portfolio becomes dollar-weighted under RAD
        # because contracts have wildly different price levels. Using
        # μ_i = 1/p_ref restores true equal-weight.
        self._ref_price = float(self._prices[0]) if self._prices[0] > 0 else 1.0

        self.history = {
            "date": [],
            "price": [],
            "reward": [],
            "daily_return": [],
            "transaction_cost": [],
            "position": [],
        }

        return self._get_state()

    def step(self, action) -> tuple[np.ndarray | None, float, bool, dict]:
        position = self._decode_action(action)

        t = self._t
        if t >= len(self._prices) - 1:
            return None, 0.0, True, {}

        price_t = self._prices[t]
        r_t = self._prices[t + 1] - price_t

        # Decision-time σ: ewm_vol[t] uses pct_change through close[t].
        ann_vol_t = self._ewm_vol[t] * math.sqrt(252)
        if np.isnan(ann_vol_t) or ann_vol_t <= 0.0:
            vol_scale = 0.0
        else:
            # 1% annualized σ floor prevents pathological scaling on
            # ultra-quiet bars; rarely binding for traded futures.
            vol_scale = self.cfg.vol_target / max(ann_vol_t, 0.01)

        position_return = vol_scale * position * r_t
        tc = self._bp * price_t * abs(vol_scale * position - self._prev_vol_scale * self._position)
        reward = (position_return - tc) / self._ref_price

        info = {
            "date": self._dates[t],
            "price": self._prices[t],
            "reward": reward,
            "daily_return": r_t,
            "transaction_cost": tc,
            "position": position,
        }
        for k, v in info.items():
            self.history[k].append(v)

        self._position = position
        self._prev_vol_scale = vol_scale
        self._t += 1

        if self._t >= len(self._prices) - 1:
            return None, reward, True, info

        return self._get_state(), reward, False, info

    def _get_state(self) -> np.ndarray:
        if self.cfg.seq_len <= 1:
            x = self._features[self._t].copy()
        else:
            start = self._t - self.cfg.seq_len + 1
            x = self._features[start : self._t + 1].copy()
        if self.cfg.state_normalize and x.ndim == 2:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sd
        return x

    def _decode_action(self, action) -> float:
        if self.cfg.action_mode == "discrete":
            return _ACTION_TO_POS[int(action)]
        return float(np.clip(action, -1.0, 1.0))
