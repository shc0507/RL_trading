"""Data pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..config import DEFAULT_OUTPUT_DIR, DEFAULT_SPLITS
from ..features import FeatureBuilder, validate_feature_frame
from ..schema import validate_canonical_bars
from .sources import BarDataSource


@dataclass(slots=True)
class BuildArtifacts:
    bars: pd.DataFrame
    features: pd.DataFrame
    bars_path: Path
    features_path: Path
    instruments_path: Path
    split_manifest_path: Path
    leakage_report_path: Path
    fetch_manifest_path: Path
    bar_quality_report_path: Path
    raw_snapshot_paths: dict[str, Path]


class MarketDataPipeline:
    """Normalize, persist, and QA market data and features."""

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        max_symbol_end_lag_business_days: int = 3,
        min_symbol_calendar_coverage: float = 0.98,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.manifest_dir = self.output_dir / "manifests"
        self.qa_dir = self.output_dir / "qa"
        self.max_symbol_end_lag_business_days = max(0, int(max_symbol_end_lag_business_days))
        self.min_symbol_calendar_coverage = float(min_symbol_calendar_coverage)
        for directory in (self.raw_dir, self.processed_dir, self.manifest_dir, self.qa_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        source: BarDataSource,
        feature_builder: FeatureBuilder,
        symbols: list[str] | tuple[str, ...],
        start_date: str,
        end_date: str,
        splits: dict[str, tuple[str, str]] | None = None,
    ) -> BuildArtifacts:
        split_config = splits or DEFAULT_SPLITS
        try:
            bars = source.fetch(symbols=symbols, start_date=start_date, end_date=end_date)
        except Exception:
            self._persist_fetch_artifacts(source)
            raise

        fetch_manifest_path, raw_snapshot_paths = self._persist_fetch_artifacts(source)

        validate_canonical_bars(bars)
        bar_quality_report = self._build_bar_quality_report(bars, start_date=start_date, end_date=end_date)
        bar_quality_report_path = self.qa_dir / "bar_quality.json"
        self._write_json_atomic(bar_quality_report_path, bar_quality_report)
        self._validate_bar_quality_report(bar_quality_report)

        features = feature_builder.transform(bars)
        validate_feature_frame(features, feature_builder.required_columns)
        leakage_report = feature_builder.check_leakage(bars)

        self._validate_split_manifest(features, split_config)

        bars_path = self.processed_dir / "bars.csv"
        features_path = self.processed_dir / "features.csv"
        instruments_path = self.processed_dir / "instruments.csv"
        split_manifest_path = self.manifest_dir / "splits.json"
        leakage_report_path = self.qa_dir / "feature_leakage.json"

        instruments = (
            bars[["symbol", "asset_class", "currency", "source"]]
            .drop_duplicates()
            .sort_values("symbol")
            .reset_index(drop=True)
        )

        self._write_frame_csv_atomic(bars_path, bars)
        self._write_frame_csv_atomic(features_path, features)
        self._write_frame_csv_atomic(instruments_path, instruments)
        self._write_json_atomic(split_manifest_path, split_config)
        self._write_json_atomic(leakage_report_path, leakage_report)

        return BuildArtifacts(
            bars=bars,
            features=features,
            bars_path=bars_path,
            features_path=features_path,
            instruments_path=instruments_path,
            split_manifest_path=split_manifest_path,
            leakage_report_path=leakage_report_path,
            fetch_manifest_path=fetch_manifest_path,
            bar_quality_report_path=bar_quality_report_path,
            raw_snapshot_paths=raw_snapshot_paths,
        )

    def _persist_fetch_artifacts(self, source: BarDataSource) -> tuple[Path, dict[str, Path]]:
        fetch_manifest = source.get_last_fetch_metadata() or {
            "source": source.__class__.__name__,
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "results": [],
        }
        fetch_manifest_path = self.manifest_dir / "fetch.json"
        self._write_json_atomic(fetch_manifest_path, fetch_manifest)

        raw_snapshot_paths: dict[str, Path] = {}
        for snapshot_key, frame in sorted(source.get_last_raw_frames().items()):
            snapshot_name = self._sanitize_filename(snapshot_key)
            snapshot_path = self.raw_dir / f"{snapshot_name}.csv"
            self._write_frame_csv_atomic(snapshot_path, frame)
            raw_snapshot_paths[snapshot_key] = snapshot_path
        return fetch_manifest_path, raw_snapshot_paths

    def _build_bar_quality_report(
        self, bars: pd.DataFrame, start_date: str, end_date: str
    ) -> dict[str, object]:
        requested_calendar = pd.DatetimeIndex(pd.bdate_range(start_date, end_date))
        observed_calendar = pd.DatetimeIndex(pd.to_datetime(bars["date"]).drop_duplicates().sort_values())
        symbol_reports = []
        for symbol, symbol_frame in bars.groupby("symbol", sort=False):
            symbol_dates = pd.DatetimeIndex(pd.to_datetime(symbol_frame["date"]).drop_duplicates().sort_values())
            common_coverage = len(symbol_dates) / len(observed_calendar) if len(observed_calendar) else 1.0
            requested_coverage = (
                len(symbol_dates.intersection(requested_calendar)) / len(requested_calendar)
                if len(requested_calendar)
                else 1.0
            )
            max_date = symbol_dates.max()
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "rows": int(len(symbol_frame)),
                    "unique_dates": int(len(symbol_dates)),
                    "min_date": self._as_date_string(symbol_dates.min()),
                    "max_date": self._as_date_string(max_date),
                    "common_calendar_coverage": float(common_coverage),
                    "requested_business_day_coverage": float(requested_coverage),
                    "end_lag_business_days": self._count_business_day_gap(max_date, pd.Timestamp(end_date)),
                }
            )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "observed_calendar_rows": int(len(observed_calendar)),
            "requested_business_days": int(len(requested_calendar)),
            "symbols": symbol_reports,
        }

    def _validate_bar_quality_report(self, report: dict[str, object]) -> None:
        failures = []
        for symbol_report in report["symbols"]:
            coverage = float(symbol_report["common_calendar_coverage"])
            if coverage < self.min_symbol_calendar_coverage:
                failures.append(
                    (
                        f"{symbol_report['symbol']} covers {coverage:.3f} of the shared calendar, "
                        f"below {self.min_symbol_calendar_coverage:.3f}"
                    )
                )
            end_lag = int(symbol_report["end_lag_business_days"])
            if end_lag > self.max_symbol_end_lag_business_days:
                failures.append(
                    (
                        f"{symbol_report['symbol']} trails the requested end date by {end_lag} "
                        f"business days, above {self.max_symbol_end_lag_business_days}"
                    )
                )
        if failures:
            raise ValueError("bar quality validation failed: " + "; ".join(failures[:3]))

    def _validate_split_manifest(
        self, features: pd.DataFrame, split_config: dict[str, tuple[str, str]]
    ) -> None:
        ranges = []
        for split_name, (start_date, end_date) in split_config.items():
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            if end < start:
                raise ValueError(f"split {split_name} has end before start")
            ranges.append((split_name, start, end))

        for index, (name_a, start_a, end_a) in enumerate(ranges):
            for name_b, start_b, end_b in ranges[index + 1 :]:
                if max(start_a, start_b) <= min(end_a, end_b):
                    raise ValueError(f"splits {name_a} and {name_b} overlap")

        min_date = pd.to_datetime(features["date"]).min()
        max_date = pd.to_datetime(features["date"]).max()
        for split_name, start_date, end_date in ranges:
            if min_date > end_date or max_date < start_date:
                raise ValueError(f"split {split_name} falls outside the feature date range")

    def _write_frame_csv_atomic(self, path: Path, frame: pd.DataFrame) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _count_business_day_gap(self, observed_end: pd.Timestamp, requested_end: pd.Timestamp) -> int:
        observed_end = pd.Timestamp(observed_end)
        requested_end = pd.Timestamp(requested_end)
        if observed_end >= requested_end:
            return 0
        return int(len(pd.bdate_range(observed_end + pd.Timedelta(days=1), requested_end)))

    def _sanitize_filename(self, value: str) -> str:
        safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
        return safe.strip("_") or "snapshot"

    def _as_date_string(self, value: pd.Timestamp | str | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")
