"""Strategy interfaces and baseline systematic signals."""

from __future__ import annotations

from dataclasses import dataclass
import math


def clamp_signal(value: float) -> float:
    """Normalize strategy outputs onto the common [-1, 1] signal scale."""

    return float(max(-1.0, min(1.0, value)))


@dataclass(slots=True)
class BaseStrategy:
    """Small interface shared by backtesting and live execution.

    Strategies emit a scalar signal in [-1, 1]:
    - `1.0` means maximum long conviction
    - `0.0` means neutral / flat
    - `-1.0` means maximum short conviction

    The portfolio layer decides how that signal becomes a position size.
    """

    name: str

    def signal(self, observation: dict[str, object]) -> float:
        raise NotImplementedError

    def act(self, observation: dict[str, object]) -> float:
        """Backward-compatible alias for older backtest and RL code paths."""

        return self.signal(observation)


@dataclass(slots=True)
class LongOnlyStrategy(BaseStrategy):
    name: str = "long_only"

    def signal(self, observation: dict[str, object]) -> float:
        return 1.0


@dataclass(slots=True)
class Sign12MStrategy(BaseStrategy):
    """Long when 12M momentum is positive, short when negative."""

    name: str = "sign_12m"

    def signal(self, observation: dict[str, object]) -> float:
        row = observation["row"]
        value = float(row.get("ret_252_raw", 0.0) or 0.0)
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0


@dataclass(slots=True)
class MACDStrategy(BaseStrategy):
    """Smoothly maps MACD into a bounded continuous conviction score."""

    name: str = "macd"

    def signal(self, observation: dict[str, object]) -> float:
        row = observation["row"]
        macd_value = float(row.get("macd_signal", 0.0) or 0.0)
        bounded = macd_value * math.exp(-(macd_value**2) / 4.0) / 0.89
        return clamp_signal(bounded)


@dataclass(slots=True)
class RSIMeanReversionStrategy(BaseStrategy):
    """Fade short-term extremes using RSI and normalized price."""

    oversold_threshold: float = 35.0
    overbought_threshold: float = 65.0
    name: str = "rsi_mean_reversion"

    def signal(self, observation: dict[str, object]) -> float:
        row = observation["row"]
        rsi_value = float(row.get("rsi_30", 50.0) or 50.0)
        norm_close = float(row.get("norm_close", 0.0) or 0.0)

        if rsi_value <= self.oversold_threshold and norm_close < 0.0:
            # The deeper the oversold reading, the stronger the rebound signal.
            return clamp_signal((self.oversold_threshold - rsi_value) / max(self.oversold_threshold, 1.0))
        if rsi_value >= self.overbought_threshold and norm_close > 0.0:
            return clamp_signal(-(rsi_value - self.overbought_threshold) / max(100.0 - self.overbought_threshold, 1.0))
        return 0.0


STRATEGY_REGISTRY = {
    "long_only": LongOnlyStrategy,
    "sign_12m": Sign12MStrategy,
    "macd": MACDStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
}


def build_strategy(name: str) -> BaseStrategy:
    normalized = name.strip().lower()
    if normalized not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError(f"unknown strategy {name}; available strategies: {available}")
    return STRATEGY_REGISTRY[normalized]()
