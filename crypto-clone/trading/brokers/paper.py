"""Local paper broker used for dry-runs and simulated execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import AccountSnapshot, Broker, OrderRequest, OrderResult, PositionSnapshot


def _base_asset_for_symbol(symbol: str, instrument_map: dict[str, dict[str, str]]) -> str:
    metadata = instrument_map.get(symbol, {})
    base_asset = metadata.get("base_asset")
    if base_asset:
        return str(base_asset).upper()
    return symbol.replace("/", "-").split("-")[0].upper()


@dataclass(slots=True)
class PaperBroker(Broker):
    """Simple spot-only broker simulation."""

    initial_cash: float = 2_000.0
    quote_currency: str = "USD"
    fee_rate_bp: float = 26.0
    instrument_map: dict[str, dict[str, str]] = field(default_factory=dict)
    name: str = "paper"
    balances: dict[str, float] = field(init=False)
    order_history: list[OrderResult] = field(init=False)

    def __post_init__(self) -> None:
        self.balances: dict[str, float] = {self.quote_currency: float(self.initial_cash)}
        self.order_history: list[OrderResult] = []

    def get_account_snapshot(self, prices: dict[str, float]) -> AccountSnapshot:
        positions = self.get_positions(prices)
        equity = self.balances.get(self.quote_currency, 0.0) + sum(
            position.market_value for position in positions.values()
        )
        return AccountSnapshot(
            timestamp=pd.Timestamp.now(tz="UTC").tz_localize(None),
            quote_currency=self.quote_currency,
            equity=float(equity),
            available_cash=float(self.balances.get(self.quote_currency, 0.0)),
            balances=dict(sorted(self.balances.items())),
        )

    def get_positions(self, prices: dict[str, float]) -> dict[str, PositionSnapshot]:
        positions: dict[str, PositionSnapshot] = {}
        for symbol, price in prices.items():
            base_asset = _base_asset_for_symbol(symbol, self.instrument_map)
            quantity = float(self.balances.get(base_asset, 0.0))
            if quantity <= 0.0:
                continue
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_price=float(price),
                market_value=float(quantity * price),
            )
        return positions

    def place_market_order(self, order: OrderRequest, price_hint: float | None = None) -> OrderResult:
        if order.quantity <= 0.0:
            raise ValueError("order quantity must be positive")
        if price_hint is None or price_hint <= 0.0:
            raise ValueError("paper broker requires a positive price_hint")

        base_asset = _base_asset_for_symbol(order.symbol, self.instrument_map)
        quote_balance = float(self.balances.get(self.quote_currency, 0.0))
        base_balance = float(self.balances.get(base_asset, 0.0))
        fee_rate = self.fee_rate_bp / 10_000.0
        notional = float(order.quantity * price_hint)
        fee = float(notional * fee_rate)

        if order.side == "buy":
            total_cost = notional + fee
            if total_cost > quote_balance + 1e-12:
                raise ValueError("insufficient cash for paper order")
            self.balances[self.quote_currency] = quote_balance - total_cost
            self.balances[base_asset] = base_balance + float(order.quantity)
        else:
            if float(order.quantity) > base_balance + 1e-12:
                raise ValueError("insufficient asset balance for paper order")
            self.balances[base_asset] = base_balance - float(order.quantity)
            self.balances[self.quote_currency] = quote_balance + notional - fee

        result = OrderResult(
            broker=self.name,
            symbol=order.symbol,
            side=order.side,
            quantity=float(order.quantity),
            average_price=float(price_hint),
            filled_notional=notional,
            fee=fee,
            status="filled",
            timestamp=pd.Timestamp.now(tz="UTC").tz_localize(None),
        )
        self.order_history.append(result)
        return result
