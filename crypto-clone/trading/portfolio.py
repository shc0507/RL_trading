"""Portfolio sizing logic shared by paper and live execution."""

from __future__ import annotations

from dataclasses import dataclass

from .strategies import clamp_signal


@dataclass(slots=True)
class PortfolioConstraints:
    """Execution-time guardrails for turning signals into orders."""

    max_symbol_weight: float = 0.30
    cash_buffer_pct: float = 0.10
    min_order_notional: float = 25.0
    allow_short: bool = False


@dataclass(slots=True)
class AllocationDecision:
    symbol: str
    signal: float
    current_weight: float
    target_weight: float
    current_value: float
    target_value: float
    delta_notional: float
    side: str | None
    quantity: float
    reason: str


class TargetAllocator:
    """Converts strategy signals into tradeable target adjustments.

    The strategy layer only says "how bullish/bearish am I?".
    This allocator answers the operational question:
    "Given account size, cash reserves, and minimum trade size, what order can I actually place?"
    """

    def __init__(self, constraints: PortfolioConstraints) -> None:
        self.constraints = constraints

    def allocate(
        self,
        symbol: str,
        signal: float,
        price: float,
        account_equity: float,
        available_cash: float,
        current_quantity: float,
    ) -> AllocationDecision:
        normalized_signal = clamp_signal(signal)
        if not self.constraints.allow_short:
            normalized_signal = max(normalized_signal, 0.0)

        current_value = current_quantity * price
        current_weight = current_value / max(account_equity, 1e-12)
        target_weight = normalized_signal * self.constraints.max_symbol_weight
        target_value = target_weight * account_equity
        delta_notional = target_value - current_value
        spendable_cash = max(available_cash - (account_equity * self.constraints.cash_buffer_pct), 0.0)
        side, quantity, reason = self._translate_to_order(
            price=price,
            delta_notional=delta_notional,
            spendable_cash=spendable_cash,
            current_quantity=current_quantity,
        )
        return AllocationDecision(
            symbol=symbol,
            signal=normalized_signal,
            current_weight=current_weight,
            target_weight=target_weight,
            current_value=current_value,
            target_value=target_value,
            delta_notional=delta_notional,
            side=side,
            quantity=quantity,
            reason=reason,
        )

    def _translate_to_order(
        self,
        price: float,
        delta_notional: float,
        spendable_cash: float,
        current_quantity: float,
    ) -> tuple[str | None, float, str]:
        absolute_notional = abs(delta_notional)
        if absolute_notional < self.constraints.min_order_notional:
            return None, 0.0, "delta below min_order_notional"

        if delta_notional > 0.0:
            buy_notional = min(delta_notional, spendable_cash)
            if buy_notional < self.constraints.min_order_notional:
                return None, 0.0, "cash buffer blocks buy"
            return "buy", float(buy_notional / price), "buy toward target weight"

        sell_quantity = min(current_quantity, absolute_notional / price)
        if sell_quantity * price < self.constraints.min_order_notional:
            return None, 0.0, "existing position too small to trim"
        return "sell", float(sell_quantity), "sell toward target weight"
