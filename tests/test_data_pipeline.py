from __future__ import annotations

import json
import tempfile
import unittest

import pandas as pd

from rl_trading.data.pipeline import MarketDataPipeline
from rl_trading.data.sources import BarDataSource
from rl_trading.features import FeatureBuilder


class StaticSource(BarDataSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def fetch(self, symbols, start_date, end_date) -> pd.DataFrame:
        self._last_fetch_metadata = {
            "source": "test_static",
            "requested_symbols": list(symbols),
            "start_date": start_date,
            "end_date": end_date,
            "results": [
                {
                    "symbol": symbol,
                    "status": "success",
                    "rows": int(len(self.frame.loc[self.frame["symbol"] == symbol])),
                }
                for symbol in symbols
            ],
        }
        self._last_raw_frames = {"static_source": self.frame.copy()}
        return self.frame.copy()


class FailingSource(BarDataSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def fetch(self, symbols, start_date, end_date) -> pd.DataFrame:
        self._last_fetch_metadata = {
            "source": "test_failing",
            "requested_symbols": list(symbols),
            "start_date": start_date,
            "end_date": end_date,
            "results": [{"symbol": symbols[0], "status": "failed", "error": "boom"}],
        }
        self._last_raw_frames = {"failing_source_partial": self.frame.copy()}
        raise RuntimeError("boom")


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
        max_date = pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            artifacts = pipeline.build(
                source=source,
                feature_builder=FeatureBuilder(),
                symbols=["AAA", "BBB"],
                start_date="2020-01-01",
                end_date=max_date,
                splits={"train": ("2020-01-01", "2020-12-31"), "val": ("2021-01-01", "2021-06-30")},
            )
            self.assertTrue(artifacts.bars_path.exists())
            self.assertTrue(artifacts.features_path.exists())
            self.assertTrue(artifacts.instruments_path.exists())
            self.assertTrue(artifacts.split_manifest_path.exists())
            self.assertTrue(artifacts.leakage_report_path.exists())
            self.assertTrue(artifacts.fetch_manifest_path.exists())
            self.assertTrue(artifacts.bar_quality_report_path.exists())
            self.assertIn("static_source", artifacts.raw_snapshot_paths)
            self.assertTrue(artifacts.raw_snapshot_paths["static_source"].exists())

            fetch_manifest = json.loads(artifacts.fetch_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(fetch_manifest["source"], "test_static")

    def test_split_overlap_is_rejected(self) -> None:
        frame = make_bar_frame(symbols=("AAA",))
        source = StaticSource(frame)
        max_date = pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            with self.assertRaises(ValueError):
                pipeline.build(
                    source=source,
                    feature_builder=FeatureBuilder(),
                    symbols=["AAA"],
                    start_date="2020-01-01",
                    end_date=max_date,
                    splits={
                        "train": ("2020-01-01", "2020-12-31"),
                        "val": ("2020-12-01", "2021-06-30"),
                    },
                )

    def test_fetch_manifest_and_raw_snapshots_persist_on_source_failure(self) -> None:
        frame = make_bar_frame(symbols=("AAA",), periods=10)
        source = FailingSource(frame)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            with self.assertRaises(RuntimeError):
                pipeline.build(
                    source=source,
                    feature_builder=FeatureBuilder(),
                    symbols=["AAA"],
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                )

            self.assertTrue((pipeline.manifest_dir / "fetch.json").exists())
            self.assertTrue((pipeline.raw_dir / "failing_source_partial.csv").exists())

    def test_bar_quality_rejects_stale_symbol_end_dates(self) -> None:
        frame = make_bar_frame()
        latest_date = pd.to_datetime(frame["date"]).max()
        stale_frame = frame.loc[
            ~((frame["symbol"] == "BBB") & (pd.to_datetime(frame["date"]) > latest_date - pd.offsets.BDay(10)))
        ].reset_index(drop=True)
        source = StaticSource(stale_frame)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = MarketDataPipeline(output_dir=tmp_dir)
            with self.assertRaises(ValueError):
                pipeline.build(
                    source=source,
                    feature_builder=FeatureBuilder(),
                    symbols=["AAA", "BBB"],
                    start_date="2020-01-01",
                    end_date=latest_date.strftime("%Y-%m-%d"),
                )


if __name__ == "__main__":
    unittest.main()
