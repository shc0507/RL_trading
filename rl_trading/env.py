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
    TRAIN_END,
    TRAIN_START,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
)
from rl_trading.features import FEATURE_COLS

# ── Split date ranges ───────────────────────────────────────────────
_SPLITS: dict[str, tuple[str, str]] = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}

# Discrete action → position mapping
_ACTION_TO_POS = {0: -1.0, 1: 0.0, 2: 1.0}

# State dimension: market features + current position
STATE_DIM = len(FEATURE_COLS) + 1


@dataclass
class EnvConfig:
    action_mode: str = "discrete"  # "discrete" or "continuous"
    cost_rate_bp: float = DEFAULT_COST_RATE_BP
    vol_target: float = DEFAULT_VOL_TARGET
    seq_len: int = OBSERVATION_WINDOW  # for LSTM sequence input


class TradingEnv:
    """Single-asset trading environment with Zhang et al. reward."""

    def __init__(self, feature_frame: pd.DataFrame, cfg: EnvConfig | None = None):
        """
        Parameters
        ----------
        feature_frame : DataFrame output of FeatureBuilder.transform(), must contain
            columns: date, symbol, close, ewm_vol, window_ready, and FEATURE_COLS.
        cfg : environment configuration.
        """
        self.full_frame = feature_frame
        self.cfg = cfg or EnvConfig()
        self._bp = self.cfg.cost_rate_bp / 10_000

        # Episode state (set in reset)
        self._data: pd.DataFrame | None = None
        self._features: np.ndarray | None = None
        self._prices: np.ndarray | None = None
        self._ewm_vol: np.ndarray | None = None
        self._dates: np.ndarray | None = None
        self._t: int = 0
        self._position: float = 0.0
        self._prev_vol_scale: float = 0.0
        self._pos_history: np.ndarray | None = None

        # Episode history
        self.history: dict[str, list] = {}

    # ── Public API ──────────────────────────────────────────────────

    def reset(
        self, symbol: str, split: str = "train",
        *, start: str | None = None, end: str | None = None,
    ) -> np.ndarray:
        """Start a new episode for *symbol* in the given split.

        Returns the first state vector (1D, length len(FEATURE_COLS)) or,
        if cfg.seq_len > 1, a 2D array of shape (seq_len, n_features).

        If *start* and *end* are provided they override *split*.
        """
        if start is not None and end is not None:
            date_start, date_end = start, end
        else:
            date_start, date_end = _SPLITS[split]
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
        self._prices = self._data["adj_close"].to_numpy(dtype=np.float64)
        self._ewm_vol = self._data["ewm_vol"].to_numpy(dtype=np.float64)
        self._dates = self._data["date"].to_numpy()

        # Start at the earliest index that allows a full sequence
        self._t = max(self.cfg.seq_len - 1, 0)
        self._position = 0.0
        self._prev_vol_scale = 0.0
        self._pos_history = np.zeros(len(self._data), dtype=np.float32)

        self.history = {
            "date": [],
            "price": [],
            "reward": [],
            "daily_return": [],
            "transaction_cost": [],
        }

        return self._get_state()

    def step(self, action) -> tuple[np.ndarray | None, float, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action : int (0/1/2) for discrete mode, float in [-1,1] for continuous.

        Returns
        -------
        (next_state, reward, done, info)
        """
        position = self._decode_action(action)

        # Current time index
        t = self._t
        # Additive return: r_t = p_{t+1} - p_t  (Zhang et al. Eq. 4)
        # We need the *next* price to compute the return earned by holding
        # a position at time t, so the last actionable step is len-2.
        if t >= len(self._prices) - 1:
            return None, 0.0, True, {}

        price_t = self._prices[t]
        r_t = self._prices[t + 1] - price_t

        # Annualized vol at time t-1 (paper uses σ_{t-1}, estimated before
        # the current return is known).  t >= seq_len-1 >= 59 from reset,
        # so t-1 >= 58 is always valid; guard included for safety.
        vol_idx = t - 1 if t >= 1 else 0
        ann_vol_t = self._ewm_vol[vol_idx] * math.sqrt(252)
        # Guard against NaN vol → don't trade; cap vol_scale to avoid
        # blow-up when vol is near zero.
        if np.isnan(ann_vol_t) or ann_vol_t <= 0.0:
            vol_scale = 0.0
        else:
            vol_scale = min(self.cfg.vol_target / ann_vol_t, 10.0)

        # Reward per Zhang et al. Eq. 4 (additive profits, σ_{t-1}):
        # R_t = (σ_tgt / σ_{t-1}) · A_t · (p_t - p_{t-1})
        #     - bp · p_t · |σ_tgt/σ_{t-1} · A_t - σ_tgt/σ_{t-2} · A_{t-1}|
        position_return = vol_scale * position * r_t
        tc = self._bp * price_t * abs(vol_scale * position - self._prev_vol_scale * self._position)
        reward = position_return - tc

        # Record history
        info = {
            "date": self._dates[t],
            "price": self._prices[t],
            "reward": reward,
            "daily_return": r_t,
            "transaction_cost": tc,
        }
        for k, v in info.items():
            self.history[k].append(v)

        # Update state — write position into current AND next slot so that
        # _get_state() at t+1 sees the current position as pos_col[-1].
        self._pos_history[self._t] = position
        if self._t + 1 < len(self._pos_history):
            self._pos_history[self._t + 1] = position
        self._position = position
        self._prev_vol_scale = vol_scale
        self._t += 1

        # Check done
        if self._t >= len(self._prices) - 1:
            return None, reward, True, info

        return self._get_state(), reward, False, info

    # ── Internals ───────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Return the current state observation (market features + position history)."""
        if self.cfg.seq_len <= 1:
            return np.append(self._features[self._t], np.float32(self._position))
        # Sequence of past states for LSTM, with actual position at each timestep
        start = self._t - self.cfg.seq_len + 1
        seq = self._features[start : self._t + 1]  # (seq_len, n_features)
        pos_col = self._pos_history[start : self._t + 1].reshape(-1, 1)
        return np.concatenate([seq, pos_col], axis=1)  # (seq_len, n_features+1)

    def _decode_action(self, action) -> float:
        if self.cfg.action_mode == "discrete":
            return _ACTION_TO_POS[int(action)]
        # Continuous: clamp to [-1, 1]
        return float(np.clip(action, -1.0, 1.0))
