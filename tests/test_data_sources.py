from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from rl_trading.data.sources import InstitutionalCsvSource, PublicDailySource


def make_yfinance_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.4, 101.4],
            "Volume": [1_000, 1_100],
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


class DataSourceTests(unittest.TestCase):
    @patch("rl_trading.data.sources.yf.download")
    def test_public_source_retries_and_records_fetch_manifest(self, mock_download) -> None:
        mock_download.side_effect = [RuntimeError("timeout"), make_yfinance_frame()]
        source = PublicDailySource(
            instrument_map={"AAA": {"asset_class": "equity_etf", "currency": "USD"}},
            max_retries=2,
            retry_delay_seconds=0,
        )

        frame = source.fetch(["AAA"], "2025-01-02", "2025-01-03")

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["symbol"].unique().tolist(), ["AAA"])
        metadata = source.get_last_fetch_metadata()
        self.assertEqual(metadata["success_count"], 1)
        self.assertEqual(metadata["failure_count"], 0)
        self.assertEqual(metadata["results"][0]["attempts"], 2)
        self.assertEqual(metadata["results"][0]["status"], "success")
        self.assertIn("public_daily_aaa", source.get_last_raw_frames())

    @patch("rl_trading.data.sources.yf.download")
    def test_public_source_can_continue_after_failed_symbol_when_not_strict(self, mock_download) -> None:
        mock_download.side_effect = [make_yfinance_frame(), pd.DataFrame()]
        source = PublicDailySource(
            instrument_map={
                "AAA": {"asset_class": "equity_etf", "currency": "USD"},
                "BBB": {"asset_class": "equity_etf", "currency": "USD"},
            },
            max_retries=1,
            retry_delay_seconds=0,
            strict=False,
        )

        frame = source.fetch(["AAA", "BBB"], "2025-01-02", "2025-01-03")

        self.assertEqual(frame["symbol"].unique().tolist(), ["AAA"])
        metadata = source.get_last_fetch_metadata()
        self.assertEqual(metadata["success_count"], 1)
        self.assertEqual(metadata["failure_count"], 1)
        statuses = {result["symbol"]: result["status"] for result in metadata["results"]}
        self.assertEqual(statuses["AAA"], "success")
        self.assertEqual(statuses["BBB"], "failed")

    def test_institutional_csv_source_can_skip_bad_file_when_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid = pd.DataFrame(
                {
                    "Date": ["2025-01-02", "2025-01-03"],
                    "Symbol": ["AAA", "AAA"],
                    "Open": [100.0, 101.0],
                    "High": [101.0, 102.0],
                    "Low": [99.0, 100.0],
                    "Close": [100.5, 101.5],
                }
            )
            invalid = pd.DataFrame({"Date": ["2025-01-02"], "Symbol": ["BBB"]})
            valid.to_csv(f"{tmp_dir}/valid.csv", index=False)
            invalid.to_csv(f"{tmp_dir}/invalid.csv", index=False)

            source = InstitutionalCsvSource(root=tmp_dir, strict=False)
            frame = source.fetch(["AAA", "BBB"], "2025-01-02", "2025-01-03")

        self.assertEqual(frame["symbol"].unique().tolist(), ["AAA"])
        metadata = source.get_last_fetch_metadata()
        self.assertEqual(metadata["success_count"], 1)
        self.assertEqual(metadata["failure_count"], 1)
        by_path = {result["path"].rsplit("/", 1)[-1]: result["status"] for result in metadata["results"]}
        self.assertEqual(by_path["valid.csv"], "success")
        self.assertEqual(by_path["invalid.csv"], "failed")


if __name__ == "__main__":
    unittest.main()
