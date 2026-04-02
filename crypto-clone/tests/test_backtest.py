from __future__ import annotations

import unittest

from trading.backtest import Backtester
from trading.env import EnvironmentConfig
from trading.features import FeatureBuilder
from trading.policies import LongOnlyPolicy, MACDPolicy, Sign12MPolicy

from test_data_pipeline import make_bar_frame


class BacktesterTests(unittest.TestCase):
    def test_backtester_emits_metrics_for_all_policies(self) -> None:
        bars = make_bar_frame(symbols=("AAA", "BBB"), periods=500)
        features = FeatureBuilder().transform(bars)
        backtester = Backtester(
            feature_frame=features,
            env_config=EnvironmentConfig(action_mode="continuous", reward_mode="zhang"),
            splits={"train": ("2020-01-01", "2022-12-31")},
        )
        for policy in (LongOnlyPolicy(), Sign12MPolicy(), MACDPolicy()):
            report = backtester.run(policy, split="train")
            self.assertIn("annualized_return", report.portfolio_metrics)
            self.assertFalse(report.symbol_metrics.empty)
            self.assertFalse(report.daily_returns.empty)
            self.assertFalse(report.trade_log.empty)


if __name__ == "__main__":
    unittest.main()
