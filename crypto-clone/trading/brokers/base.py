"""Broker interfaces shared by paper and live trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

OrderSide = Literal["buy", "sell"]


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str = "market"
    client_order_id: str | None = None


@dataclass(slots=True)
class OrderResult:
    broker: str
    symbol: str
    side: OrderSide
    quantity: float
    average_price: float
    filled_notional: float
    fee: float
    status: str
    timestamp: pd.Timestamp
    raw_response: dict[str, object] | None = None


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    market_price: float
    market_value: float


@dataclass(slots=True)
class AccountSnapshot:
    timestamp: pd.Timestamp
    quote_currency: str
    equity: float
    available_cash: float
    balances: dict[str, float] = field(default_factory=dict)


class Broker(ABC):
    """Abstract broker used by the live runner."""

    name: str = "abstract"
    quote_currency: str = "USD"

    @abstractmethod
    def get_account_snapshot(self, prices: dict[str, float]) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self, prices: dict[str, float]) -> dict[str, PositionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, order: OrderRequest, price_hint: float | None = None) -> OrderResult:
        raise NotImplementedError
