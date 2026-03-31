"""Data pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

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


class MarketDataPipeline:
    """Normalize, persist, and QA market data and features."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.manifest_dir = self.output_dir / "manifests"
        self.qa_dir = self.output_dir / "qa"
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
        bars = source.fetch(symbols=symbols, start_date=start_date, end_date=end_date)
        validate_canonical_bars(bars)

        features = feature_builder.transform(bars)
        validate_feature_frame(features, feature_builder.required_columns)
        leakage_report = feature_builder.check_leakage(bars)

        bars_path = self.processed_dir / "bars.csv"
        features_path = self.processed_dir / "features.csv"
        instruments_path = self.processed_dir / "instruments.csv"
        split_manifest_path = self.manifest_dir / "splits.json"
        leakage_report_path = self.qa_dir / "feature_leakage.json"

        bars.to_csv(bars_path, index=False)
        features.to_csv(features_path, index=False)
        instruments = (
            bars[["symbol", "asset_class", "currency", "source"]]
            .drop_duplicates()
            .sort_values("symbol")
            .reset_index(drop=True)
        )
        instruments.to_csv(instruments_path, index=False)
        split_manifest_path.write_text(json.dumps(split_config, indent=2))
        leakage_report_path.write_text(json.dumps(leakage_report, indent=2))

        self._validate_split_manifest(features, split_config)

        return BuildArtifacts(
            bars=bars,
            features=features,
            bars_path=bars_path,
            features_path=features_path,
            instruments_path=instruments_path,
            split_manifest_path=split_manifest_path,
            leakage_report_path=leakage_report_path,
        )

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
