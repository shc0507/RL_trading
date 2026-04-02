from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading.pairs import PairsConfig, PairsTradingResearcher


def make_cointegrated_pair_bars(periods: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    base_returns = rng.normal(0.0005, 0.01, periods)
    base_price = 100.0 * np.exp(np.cumsum(base_returns))
    spread_noise = rng.normal(0.0, 0.01, periods)
    paired_price = np.exp((np.log(base_price) - 0.25 - spread_noise) / 1.15)

    frames = []
    for symbol, series in (("AAA", base_price), ("BBB", paired_price)):
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "asset_class": "equity_etf",
                "open": series,
                "high": series * 1.001,
                "low": series * 0.999,
                "close": series,
                "adj_close": series,
                "volume": 100_000,
                "source": "test",
                "currency": "USD",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class PairsTradingResearcherTests(unittest.TestCase):
    def test_cointegration_and_threshold_selection(self) -> None:
        bars = make_cointegrated_pair_bars()
        researcher = PairsTradingResearcher(
            bar_frame=bars,
            config=PairsConfig(
                lookback=20,
                cost_rate_bp=5.0,
                entry_thresholds=(1.0, 1.5, 2.0),
                exit_thresholds=(0.0, 0.5),
                cv_folds=3,
                min_train_size=120,
            ),
        )

        result = researcher.run(
            symbol_x="AAA",
            symbol_y="BBB",
            train_start="2020-01-01",
            train_end="2020-11-30",
            test_start="2020-12-01",
            test_end="2021-05-31",
        )

        self.assertTrue(result.train_cointegration.is_cointegrated)
        self.assertTrue(result.is_tradeable)
        self.assertIsNotNone(result.threshold_selection.best_entry_threshold)
        self.assertIsNotNone(result.threshold_selection.best_exit_threshold)
        self.assertIsNotNone(result.test_report)
        self.assertFalse(result.threshold_selection.cv_results.empty)
        self.assertIn("annualized_return", result.test_report.metrics)


if __name__ == "__main__":
    unittest.main()
