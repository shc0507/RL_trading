"""Baseline trading policies."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class BasePolicy:
    name: str

    def act(self, observation: dict[str, object]) -> float:
        raise NotImplementedError


@dataclass(slots=True)
class LongOnlyPolicy(BasePolicy):
    name: str = "long_only"

    def act(self, observation: dict[str, object]) -> float:
        return 1.0


@dataclass(slots=True)
class Sign12MPolicy(BasePolicy):
    name: str = "sign_12m"

    def act(self, observation: dict[str, object]) -> float:
        row = observation["row"]
        value = float(row.get("ret_252_raw", 0.0) or 0.0)
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0


@dataclass(slots=True)
class MACDPolicy(BasePolicy):
    name: str = "macd"

    def act(self, observation: dict[str, object]) -> float:
        row = observation["row"]
        macd_value = float(row.get("macd_signal", 0.0) or 0.0)
        signal = macd_value * math.exp(-(macd_value**2) / 4.0) / 0.89
        return float(max(-1.0, min(1.0, signal)))
