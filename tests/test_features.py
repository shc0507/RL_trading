from __future__ import annotations

import unittest

import pandas as pd

from rl_trading.features import FeatureBuilder

try:
    from test_data_pipeline import make_bar_frame
except ModuleNotFoundError:  # pragma: no cover - supports module-style unittest invocation
    from tests.test_data_pipeline import make_bar_frame


class FeatureTests(unittest.TestCase):
    def test_feature_builder_produces_ready_rows_without_nans(self) -> None:
        frame = make_bar_frame(symbols=("AAA",), periods=500)
        features = FeatureBuilder().transform(frame)
        ready = features.loc[features["window_ready"]]
        self.assertFalse(ready.empty)
        self.assertFalse(ready[list(FeatureBuilder().feature_columns) + ["ewm_vol_60"]].isna().any().any())

    def test_rsi_stays_in_expected_range(self) -> None:
        frame = make_bar_frame(symbols=("AAA",), periods=500)
        features = FeatureBuilder().transform(frame)
        valid = features["rsi_30"].dropna()
        self.assertTrue(((valid >= 0.0) & (valid <= 100.0)).all())

    def test_feature_leakage_check_passes(self) -> None:
        frame = make_bar_frame(symbols=("AAA",), periods=500)
        report = FeatureBuilder().check_leakage(frame)
        self.assertTrue(all(entry["passed"] for entry in report["AAA"]))


if __name__ == "__main__":
    unittest.main()
