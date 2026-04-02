"""Daily live trading runner for paper and live broker execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import pandas as pd

from .brokers import Broker, KrakenBroker, KrakenRESTClient, OrderRequest, OrderResult, PaperBroker
from .config import DEFAULT_OUTPUT_DIR, get_default_symbols, get_instrument_map
from .data.sources import PublicDailySource
from .features import FeatureBuilder
from .portfolio import AllocationDecision, PortfolioConstraints, TargetAllocator
from .strategies import BaseStrategy, build_strategy


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


@dataclass(slots=True)
class LiveTradingConfig:
    market: str = "crypto"
    symbols: tuple[str, ...] = get_default_symbols("crypto")
    lookback_days: int = 450
    poll_seconds: int = 300
    max_symbol_weight: float = 0.30
    cash_buffer_pct: float = 0.10
    min_order_notional: float = 25.0
    allow_short: bool = False
    dry_run: bool = True
    initial_cash: float = 2_000.0
    fee_rate_bp: float = 26.0
    output_dir: Path = DEFAULT_OUTPUT_DIR / "live"

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "LiveTradingConfig":
        raw = dict(payload)
        if "symbols" in raw:
            raw["symbols"] = tuple(str(symbol) for symbol in raw["symbols"])
        if "output_dir" in raw:
            raw["output_dir"] = Path(str(raw["output_dir"]))
        return cls(**raw)


@dataclass(slots=True)
class TradeDecision:
    symbol: str
    signal: float
    current_weight: float
    target_weight: float
    price: float
    current_value: float
    target_value: float
    delta_notional: float
    side: str | None
    quantity: float
    reason: str


@dataclass(slots=True)
class CycleReport:
    timestamp: pd.Timestamp
    strategy_name: str
    account_equity: float
    available_cash: float
    decisions: list[TradeDecision]
    orders: list[OrderResult]


class LiveTrader:
    """Fetches data, scores the strategy, sizes trades, and sends orders."""

    def __init__(
        self,
        source: PublicDailySource,
        feature_builder: FeatureBuilder,
        broker: Broker,
        instrument_map: dict[str, dict[str, str]],
        config: LiveTradingConfig,
    ) -> None:
        self.source = source
        self.feature_builder = feature_builder
        self.broker = broker
        self.instrument_map = instrument_map
        self.config = config
        self.allocator = TargetAllocator(
            PortfolioConstraints(
                max_symbol_weight=config.max_symbol_weight,
                cash_buffer_pct=config.cash_buffer_pct,
                min_order_notional=config.min_order_notional,
                allow_short=config.allow_short,
            )
        )

    def run_cycle(self, strategy: BaseStrategy) -> CycleReport:
        bars = self._fetch_recent_bars()
        features = self.feature_builder.transform(bars)
        latest_rows: dict[str, pd.Series] = {}
        prices: dict[str, float] = {}
        for symbol in self.config.symbols:
            symbol_frame = features.loc[(features["symbol"] == symbol) & (features["window_ready"])].copy()
            if symbol_frame.empty:
                raise ValueError(f"no feature-ready rows found for {symbol}")
            latest_row = symbol_frame.iloc[-1]
            latest_rows[symbol] = latest_row
            prices[symbol] = float(latest_row["adj_close"])

        account = self.broker.get_account_snapshot(prices)
        positions = self.broker.get_positions(prices)
        account_equity = max(float(account.equity), 1e-12)
        available_cash = float(account.available_cash)

        decisions: list[TradeDecision] = []
        orders: list[OrderResult] = []
        for symbol in self.config.symbols:
            row = latest_rows[symbol]
            price = prices[symbol]
            position = positions.get(symbol)
            current_quantity = 0.0 if position is None else float(position.quantity)
            observation = self._build_observation(symbol, features, row, current_quantity, account_equity, price)
            strategy_signal = strategy.signal(observation)
            allocation = self.allocator.allocate(
                symbol=symbol,
                signal=strategy_signal,
                price=price,
                account_equity=account_equity,
                available_cash=available_cash,
                current_quantity=current_quantity,
            )
            decision = self._make_trade_decision(price=price, allocation=allocation)
            decisions.append(decision)

            if decision.side is None or decision.quantity <= 0.0:
                continue
            if not self.config.dry_run:
                order = self.broker.place_market_order(
                    OrderRequest(symbol=symbol, side=decision.side, quantity=decision.quantity),
                    price_hint=price,
                )
                orders.append(order)
                if decision.side == "buy":
                    available_cash = max(available_cash - order.filled_notional - order.fee, 0.0)
                else:
                    available_cash += order.filled_notional - order.fee
            elif decision.side == "buy":
                available_cash = max(available_cash - (decision.quantity * price), 0.0)
            else:
                available_cash += decision.quantity * price

        return CycleReport(
            timestamp=_utc_now_naive(),
            strategy_name=strategy.name,
            account_equity=float(account.equity),
            available_cash=float(account.available_cash),
            decisions=decisions,
            orders=orders,
        )

    def persist_report(self, report: CycleReport) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": report.timestamp,
            "strategy_name": report.strategy_name,
            "account_equity": report.account_equity,
            "available_cash": report.available_cash,
            "decisions": [asdict(decision) for decision in report.decisions],
            "orders": [asdict(order) for order in report.orders],
        }
        latest_path = output_dir / "latest_cycle.json"
        archive_path = output_dir / f"cycle_{report.timestamp.strftime('%Y%m%dT%H%M%S')}.json"
        serialized = json.dumps(payload, default=_json_default, indent=2)
        latest_path.write_text(serialized)
        archive_path.write_text(serialized)
        return latest_path

    def run_forever(self, strategy: BaseStrategy) -> None:
        while True:
            report = self.run_cycle(strategy)
            self.persist_report(report)
            print(
                f"{report.timestamp.isoformat()} strategy={report.strategy_name} "
                f"equity={report.account_equity:.2f} decisions={len(report.decisions)} orders={len(report.orders)}"
            )
            time.sleep(self.config.poll_seconds)

    def _fetch_recent_bars(self) -> pd.DataFrame:
        end_date = _utc_now_naive().strftime("%Y-%m-%d")
        start_date = (_utc_now_naive() - pd.Timedelta(days=self.config.lookback_days)).strftime("%Y-%m-%d")
        return self.source.fetch(
            symbols=self.config.symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def _build_observation(
        self,
        symbol: str,
        features: pd.DataFrame,
        latest_row: pd.Series,
        current_quantity: float,
        account_equity: float,
        price: float,
    ) -> dict[str, object]:
        symbol_frame = features.loc[(features["symbol"] == symbol) & (features["window_ready"])].copy()
        window = symbol_frame.loc[:, self.feature_builder.feature_columns].tail(self.feature_builder.observation_window)
        return {
            "symbol": symbol,
            "split": "live",
            "date": pd.Timestamp(latest_row["date"]),
            # We pass weight rather than raw units so strategies can stay market-agnostic.
            "position": (current_quantity * price) / max(account_equity, 1e-12),
            "window": window.to_numpy(dtype=float),
            "features": {column: float(latest_row[column]) for column in self.feature_builder.feature_columns},
            "row": latest_row.to_dict(),
        }

    def _make_trade_decision(self, price: float, allocation: AllocationDecision) -> TradeDecision:
        return TradeDecision(
            symbol=allocation.symbol,
            signal=allocation.signal,
            current_weight=allocation.current_weight,
            target_weight=allocation.target_weight,
            price=price,
            current_value=allocation.current_value,
            target_value=allocation.target_value,
            delta_notional=allocation.delta_notional,
            side=allocation.side,
            quantity=allocation.quantity,
            reason=allocation.reason,
        )


def load_live_config(path: str | Path | None) -> LiveTradingConfig:
    if path is None:
        return LiveTradingConfig()
    payload = json.loads(Path(path).read_text())
    return LiveTradingConfig.from_mapping(payload)


def build_broker(
    broker_name: str,
    config: LiveTradingConfig,
    instrument_map: dict[str, dict[str, str]],
) -> Broker:
    if broker_name == "paper":
        return PaperBroker(
            initial_cash=config.initial_cash,
            quote_currency="USD",
            fee_rate_bp=config.fee_rate_bp,
            instrument_map=instrument_map,
        )
    if broker_name == "kraken":
        api_key = os.environ.get("KRAKEN_API_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET")
        if not api_key or not api_secret:
            raise EnvironmentError("KRAKEN_API_KEY and KRAKEN_API_SECRET are required for broker=kraken")
        return KrakenBroker(
            client=KrakenRESTClient(api_key=api_key, api_secret=api_secret),
            quote_currency="USD",
            instrument_map=instrument_map,
        )
    raise KeyError(f"unknown broker {broker_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strategy-driven daily crypto trading loop.")
    parser.add_argument("--config", default="")
    parser.add_argument(
        "--strategy",
        choices=("long_only", "sign_12m", "macd", "rsi_mean_reversion"),
        default="macd",
    )
    parser.add_argument("--broker", choices=("paper", "kraken"), default="paper")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_live_config(args.config or None)
    config.dry_run = not args.execute
    instrument_map = get_instrument_map(config.market)
    source = PublicDailySource(instrument_map=instrument_map)
    broker = build_broker(args.broker, config, instrument_map)
    trader = LiveTrader(
        source=source,
        feature_builder=FeatureBuilder(),
        broker=broker,
        instrument_map=instrument_map,
        config=config,
    )
    strategy = build_strategy(args.strategy)

    if args.once:
        report = trader.run_cycle(strategy)
        path = trader.persist_report(report)
        print(f"wrote {path}")
        return

    trader.run_forever(strategy)


if __name__ == "__main__":
    main()
