from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from trading.data.pipeline import MarketDataPipeline
from trading.data.sources import BarDataSource
from trading.features import FeatureBuilder


class StaticSource(BarDataSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def fetch(self, symbols, start_date, end_date) -> pd.DataFrame:
        return self.frame.copy()


def make_bar_frame(symbols=("AAA", "BBB"), periods=400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    frames = []
    for idx, symbol in enumerate(symbols, start=1):
        price = pd.Series(range(periods), dtype=float) * 0.1 + (100 + idx)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "asset_class": "equity_etf",
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000,
                "source": "test",
                "currency": "USD",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class PipelineTests(unittest.TestCase):
    def test_pipeline_persists_artifacts_and_split_manifest(self) -> None:
        frame = make_bar_frame()
        source = StaticSource(frame)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            artifacts = pipeline.build(
                source=source,
                feature_builder=FeatureBuilder(),
                symbols=["AAA", "BBB"],
                start_date="2020-01-01",
                end_date="2021-12-31",
                splits={"train": ("2020-01-01", "2020-12-31"), "val": ("2021-01-01", "2021-06-30")},
            )
            self.assertTrue(artifacts.bars_path.exists())
            self.assertTrue(artifacts.features_path.exists())
            self.assertTrue(artifacts.instruments_path.exists())
            self.assertTrue(artifacts.split_manifest_path.exists())
            self.assertTrue(artifacts.leakage_report_path.exists())

    def test_split_overlap_is_rejected(self) -> None:
        frame = make_bar_frame(symbols=("AAA",))
        source = StaticSource(frame)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            with self.assertRaises(ValueError):
                pipeline.build(
                    source=source,
                    feature_builder=FeatureBuilder(),
                    symbols=["AAA"],
                    start_date="2020-01-01",
                    end_date="2021-12-31",
                    splits={
                        "train": ("2020-01-01", "2020-12-31"),
                        "val": ("2020-12-01", "2021-06-30"),
                    },
                )


if __name__ == "__main__":
    unittest.main()
