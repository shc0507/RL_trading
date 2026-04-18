from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rl_trading.evaluate import _filter_full_history_universe, _portfolio_vol_scale


class EvaluateHelpersTest(unittest.TestCase):
    def test_filter_full_history_universe_excludes_partial_symbol(self) -> None:
        folds = [
            {
                "train": ("2020-01-01", "2020-01-03"),
                "val": ("2020-01-06", "2020-01-08"),
                "test": ("2020-01-09", "2020-01-13"),
            },
            {
                "train": ("2020-01-01", "2020-01-08"),
                "val": ("2020-01-09", "2020-01-10"),
                "test": ("2020-01-13", "2020-01-15"),
            },
        ]
        feat = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2020-01-01", "2020-01-02", "2020-01-03",
                        "2020-01-06", "2020-01-07", "2020-01-08",
                        "2020-01-09", "2020-01-10", "2020-01-13",
                        "2020-01-14", "2020-01-15",
                    ] * 2
                ),
                "symbol": ["OLD"] * 11 + ["LATE"] * 11,
                "window_ready": [True] * 14 + [False] * 3 + [True] * 5,
            }
        )
        universe = [
            {"symbol": "OLD", "asset_class": "commodity"},
            {"symbol": "LATE", "asset_class": "commodity"},
        ]

        eligible, exclusions = _filter_full_history_universe(
            universe,
            feat,
            folds,
            min_rows=2,
        )

        self.assertEqual([entry["symbol"] for entry in eligible], ["OLD"])
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0]["symbol"], "LATE")
        self.assertEqual(exclusions[0]["fold_idx"], 1)
        self.assertEqual(exclusions[0]["split"], "val")

    def test_portfolio_vol_scale_caps_near_zero_vol(self) -> None:
        raw = np.full(80, 1e-5, dtype=np.float64)

        scaled = _portfolio_vol_scale(raw, vol_target=0.15)

        np.testing.assert_allclose(scaled[:60], raw[:60])
        self.assertLessEqual(float(np.max(np.abs(scaled))), 1e-4 + 1e-12)
        self.assertAlmostEqual(float(scaled[60]), 1e-4, places=12)


if __name__ == "__main__":
    unittest.main()
