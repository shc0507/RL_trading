from __future__ import annotations

import unittest

import pandas as pd

from rl_trading.zhang_asset_class_experiment import (
    ASSET_CLASS_SPECS,
    build_window_splits,
    eligible_symbols_for_windows,
    format_cost_tag,
    resolve_training_budget,
    zhang_splits,
    zhang_windows,
)


class ZhangAssetClassExperimentTests(unittest.TestCase):
    def test_asset_class_specs_include_fixed_income_and_commodities(self) -> None:
        self.assertIn("fixed_income", ASSET_CLASS_SPECS)
        self.assertIn("commodities", ASSET_CLASS_SPECS)
        self.assertGreater(len(ASSET_CLASS_SPECS["fixed_income"].symbols), 0)
        self.assertGreater(len(ASSET_CLASS_SPECS["commodities"].symbols), 0)

    def test_resolve_training_budget_defaults_to_twenty_passes(self) -> None:
        total_timesteps, eval_frequency = resolve_training_budget(base_training_steps=250, total_timesteps=None)
        self.assertEqual(total_timesteps, 5000)
        self.assertEqual(eval_frequency, 250)

    def test_format_cost_tag(self) -> None:
        self.assertEqual(format_cost_tag(5.0), "5bp")
        self.assertEqual(format_cost_tag(2.5), "2p5bp")

    def test_eligible_symbols_for_windows_filters_incomplete_symbols(self) -> None:
        dates = pd.date_range("2007-03-30", periods=5000, freq="B")
        frame = pd.DataFrame(
            {
                "date": list(dates) + list(dates[4000:]),
                "symbol": ["AAA"] * len(dates) + ["BBB"] * len(dates[4000:]),
                "window_ready": [True] * len(dates) + [True] * len(dates[4000:]),
            }
        )
        eligible, dropped = eligible_symbols_for_windows(
            frame,
            symbols=["AAA", "BBB"],
            windows=zhang_windows(),
            splits=zhang_splits(),
        )
        self.assertEqual(eligible, ["AAA"])
        self.assertIn("BBB", dropped)

    def test_build_window_splits_still_creates_non_overlapping_ranges(self) -> None:
        dates = pd.date_range("2007-03-30", periods=100, freq="B")
        frame = pd.DataFrame({"date": [str(date.date()) for date in dates]})
        splits = build_window_splits(
            frame,
            train_end=str(dates[79].date()),
            test_start=str(dates[80].date()),
            test_end=str(dates[-1].date()),
        )
        self.assertLess(pd.Timestamp(splits["train_fit"][1]), pd.Timestamp(splits["train_val"][0]))
        self.assertEqual(splits["test"], (str(dates[80].date()), str(dates[-1].date())))


if __name__ == "__main__":
    unittest.main()
