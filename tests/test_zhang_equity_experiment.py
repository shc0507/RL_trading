from __future__ import annotations

import unittest

import pandas as pd

from rl_trading.backtest import EvalReport
from rl_trading.zhang_equity_experiment import (
    aggregate_eval_reports,
    build_window_splits,
    count_training_steps,
    resolve_training_budget,
    zhang_equity_splits,
    zhang_equity_windows,
)


class ZhangEquityExperimentTests(unittest.TestCase):
    def test_split_windows_match_expected_structure(self) -> None:
        splits = zhang_equity_splits()
        windows = zhang_equity_windows()
        self.assertIn("train_pre_2011", splits)
        self.assertIn("test_2016_2019", splits)
        self.assertEqual(windows[0], ("2011_2015", "train_pre_2011", "test_2011_2015"))

    def test_aggregate_eval_reports_combines_daily_returns(self) -> None:
        report_one = EvalReport(
            policy_name="demo",
            split="test_a",
            portfolio_metrics={"annualized_return": 0.1},
            symbol_metrics=pd.DataFrame([{"symbol": "AAA", "annualized_return": 0.1}]),
            daily_returns=pd.DataFrame(
                [
                    {"date": pd.Timestamp("2011-01-03"), "portfolio_return": 0.01, "avg_turnover": 0.1, "total_cost": 1.0},
                    {"date": pd.Timestamp("2011-01-04"), "portfolio_return": -0.02, "avg_turnover": 0.2, "total_cost": 2.0},
                ]
            ),
            trade_log=pd.DataFrame(
                [
                    {"date": pd.Timestamp("2011-01-03"), "symbol": "AAA"},
                    {"date": pd.Timestamp("2011-01-04"), "symbol": "AAA"},
                ]
            ),
        )
        report_two = EvalReport(
            policy_name="demo",
            split="test_b",
            portfolio_metrics={"annualized_return": 0.2},
            symbol_metrics=pd.DataFrame([{"symbol": "AAA", "annualized_return": 0.2}]),
            daily_returns=pd.DataFrame(
                [
                    {"date": pd.Timestamp("2016-01-05"), "portfolio_return": 0.03, "avg_turnover": 0.3, "total_cost": 3.0},
                ]
            ),
            trade_log=pd.DataFrame(
                [
                    {"date": pd.Timestamp("2016-01-05"), "symbol": "AAA"},
                ]
            ),
        )

        combined = aggregate_eval_reports("demo", [report_one, report_two], "combined")
        self.assertEqual(combined.policy_name, "demo")
        self.assertEqual(combined.split, "combined")
        self.assertEqual(len(combined.daily_returns), 3)
        self.assertIn("annualized_return", combined.portfolio_metrics)
        self.assertEqual(len(combined.trade_log), 3)

    def test_build_window_splits_creates_train_val_test_ranges(self) -> None:
        dates = pd.date_range("2007-03-30", periods=100, freq="B")
        frame = pd.DataFrame({"date": dates})
        splits = build_window_splits(
            frame,
            train_end=str(dates[79].date()),
            test_start=str(dates[80].date()),
            test_end=str(dates[-1].date()),
        )
        self.assertEqual(set(splits.keys()), {"train_fit", "train_val", "test"})
        self.assertLess(pd.Timestamp(splits["train_fit"][1]), pd.Timestamp(splits["train_val"][0]))
        self.assertEqual(splits["test"], (str(dates[80].date()), str(dates[-1].date())))

    def test_build_window_splits_accepts_string_dates_and_count_training_steps(self) -> None:
        dates = pd.date_range("2007-03-30", periods=80, freq="B")
        per_symbol_window_ready = [False] + [True] * (len(dates) - 1)
        frame = pd.DataFrame(
            {
                "date": [str(date.date()) for date in dates] * 2,
                "symbol": ["AAA"] * len(dates) + ["BBB"] * len(dates),
                "window_ready": per_symbol_window_ready * 2,
            }
        )
        splits = build_window_splits(
            frame,
            train_end=str(dates[59].date()),
            test_start=str(dates[60].date()),
            test_end=str(dates[-1].date()),
        )
        steps = count_training_steps(frame, ["AAA", "BBB"], splits["train_fit"])
        self.assertGreater(steps, 0)
        self.assertLess(pd.Timestamp(splits["train_fit"][1]), pd.Timestamp(splits["train_val"][0]))

    def test_resolve_training_budget_uses_epoch_multiplier_by_default(self) -> None:
        total_timesteps, eval_frequency = resolve_training_budget(base_training_steps=300, total_timesteps=None)
        self.assertEqual(total_timesteps, 6000)
        self.assertEqual(eval_frequency, 300)

    def test_resolve_training_budget_respects_manual_override(self) -> None:
        total_timesteps, eval_frequency = resolve_training_budget(
            base_training_steps=300,
            total_timesteps=1500,
            epoch_multiplier=20,
        )
        self.assertEqual(total_timesteps, 1500)
        self.assertEqual(eval_frequency, 75)


if __name__ == "__main__":
    unittest.main()
