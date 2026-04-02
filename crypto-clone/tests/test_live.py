from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from trading.config import DEFAULT_CRYPTO_INSTRUMENT_MAP
from trading.data.sources import BarDataSource
from trading.features import FeatureBuilder
from trading.live import LiveTrader, LiveTradingConfig
from trading.brokers import PaperBroker
from trading.strategies import LongOnlyStrategy


class StaticSource(BarDataSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def fetch(self, symbols, start_date, end_date) -> pd.DataFrame:
        return self.frame.loc[self.frame["symbol"].isin(symbols)].copy()


def make_crypto_bars(symbols=("BTC-USD", "ETH-USD"), periods=500) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    frames = []
    for idx, symbol in enumerate(symbols, start=1):
        price = pd.Series(range(periods), dtype=float) * (10 + idx) + (1_000 + 100 * idx)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "asset_class": "crypto_spot",
                "open": price,
                "high": price + 5,
                "low": price - 5,
                "close": price,
                "adj_close": price,
                "volume": 10_000,
                "source": "test",
                "currency": "USD",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class LiveTraderTests(unittest.TestCase):
    def test_run_cycle_creates_buy_decisions(self) -> None:
        frame = make_crypto_bars()
        source = StaticSource(frame)
        broker = PaperBroker(initial_cash=2_000.0, instrument_map=DEFAULT_CRYPTO_INSTRUMENT_MAP)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trader = LiveTrader(
                source=source,
                feature_builder=FeatureBuilder(),
                broker=broker,
                instrument_map=DEFAULT_CRYPTO_INSTRUMENT_MAP,
                config=LiveTradingConfig(
                    symbols=("BTC-USD", "ETH-USD"),
                    output_dir=tmp_dir,
                    dry_run=False,
                    max_symbol_weight=0.25,
                    min_order_notional=50.0,
                ),
            )
            report = trader.run_cycle(LongOnlyStrategy())
            self.assertEqual(len(report.orders), 2)
            self.assertTrue(all(order.side == "buy" for order in report.orders))
            self.assertTrue(all(decision.target_weight > 0.0 for decision in report.decisions))


if __name__ == "__main__":
    unittest.main()
