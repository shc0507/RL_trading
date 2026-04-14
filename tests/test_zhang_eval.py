from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from rl_trading.asset_groups import canonical_asset_group
from rl_trading.rl.zhang_eval import ZhangEvalConfig, ZhangEvaluator


def make_trade_logs() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=5)
    next_dates = pd.bdate_range("2022-01-04", periods=5)
    returns = {
        ("dqn", "AAA"): [0.0100, -0.0200, 0.0300, -0.0100, 0.0200],
        ("dqn", "BBB"): [0.0060, -0.0150, 0.0250, -0.0060, 0.0140],
        ("long_only", "AAA"): [0.0080, 0.0010, 0.0120, -0.0040, 0.0100],
        ("long_only", "BBB"): [0.0050, 0.0020, 0.0100, -0.0030, 0.0080],
    }
    turnovers = {
        "AAA": [0.40, 0.50, 0.30, 0.60, 0.20],
        "BBB": [0.20, 0.25, 0.35, 0.30, 0.15],
    }
    rows: list[dict[str, object]] = []
    for (policy, symbol), series in returns.items():
        for index, trade_return in enumerate(series):
            scaled_turnover = turnovers[symbol][index]
            turnover = scaled_turnover * 0.8
            price_now = 100.0 + index
            zhang_cost_return = 0.0005 * scaled_turnover
            raw_cost_return = 0.0005 * turnover
            rows.append(
                {
                    "date": dates[index],
                    "next_date": next_dates[index],
                    "symbol": symbol,
                    "split": "test",
                    "scaled_position": 0.5 if trade_return >= 0.0 else -0.5,
                    "turnover": turnover,
                    "scaled_turnover": scaled_turnover,
                    "raw_return": trade_return + zhang_cost_return - raw_cost_return,
                    "raw_cost_return": raw_cost_return,
                    "raw_cost": raw_cost_return * price_now,
                    "zhang_return": trade_return,
                    "zhang_cost_return": zhang_cost_return,
                    "zhang_cost": zhang_cost_return * price_now,
                    "trade_return": trade_return * price_now,
                    "pre_cost_trade_return": (trade_return + zhang_cost_return) * price_now,
                    "trade_cost": zhang_cost_return * price_now,
                    "price_now": price_now,
                    "policy": policy,
                }
            )
    return pd.DataFrame(rows)


def make_instruments() -> pd.DataFrame:
    instruments = pd.DataFrame(
        [
            {"symbol": "AAA", "asset_class": "equity_etf"},
            {"symbol": "BBB", "asset_class": "fx"},
        ]
    )
    instruments["asset_group"] = instruments["asset_class"].map(canonical_asset_group)
    return instruments


class ZhangEvalTests(unittest.TestCase):
    def test_evaluator_uses_additive_cumulative_returns_and_emits_groups(self) -> None:
        evaluator = ZhangEvaluator(
            trade_logs=make_trade_logs(),
            instruments=make_instruments(),
            config=ZhangEvalConfig(
                target_vol=0.05,
                cost_grid_bp=(1.0, 10.0),
                split="test",
                vol_span=2,
            ),
        )
        results = evaluator.evaluate()

        daily_unscaled = results["daily_portfolios_unscaled"]
        equity_dqn = daily_unscaled.loc[
            (daily_unscaled["policy"] == "dqn")
            & (daily_unscaled["split"] == "test")
            & (daily_unscaled["asset_group"] == "equity_index")
        ].reset_index(drop=True)
        expected_cumulative = equity_dqn["portfolio_return"].fillna(0.0).cumsum()
        pd.testing.assert_series_equal(
            equity_dqn["cumulative_trade_return"],
            expected_cumulative,
            check_names=False,
        )

        self.assertSetEqual(
            set(daily_unscaled["asset_group"].unique().tolist()),
            {"all", "equity_index", "fx"},
        )
        instrument_groups = results["instrument_groups"].set_index("symbol")["asset_group"].to_dict()
        self.assertEqual(instrument_groups["AAA"], "equity_index")

        contract_daily = results["contract_daily_returns"]
        self.assertTrue(
            {"trade_return", "trade_cost", "scaled_turnover", "pre_cost_trade_return", "scaled_position"}.issubset(
                contract_daily.columns
            )
        )

    def test_portfolio_scaling_and_cost_sweep_outputs_change_with_targeting(self) -> None:
        evaluator = ZhangEvaluator(
            trade_logs=make_trade_logs(),
            instruments=make_instruments(),
            config=ZhangEvalConfig(
                target_vol=0.05,
                cost_grid_bp=(1.0, 10.0, 25.0),
                split="test",
                vol_span=2,
            ),
        )
        results = evaluator.evaluate()

        daily_unscaled = results["daily_portfolios_unscaled"]
        daily_scaled = results["daily_portfolios_scaled"]
        merged = daily_unscaled.merge(
            daily_scaled,
            on=["policy", "split", "asset_group", "date"],
            suffixes=("_unscaled", "_scaled"),
        )
        differences = (merged["portfolio_return_scaled"] - merged["portfolio_return_unscaled"]).abs()
        self.assertGreater(float(differences.max()), 0.0)

        summary_scaled = results["summary_scaled"]
        summary_unscaled = results["summary_unscaled"]
        self.assertFalse(summary_scaled.equals(summary_unscaled))

        cost_sweep = results["cost_sweep"]
        self.assertEqual(len(cost_sweep), 2 * 3 * 3)
        dqn_equity = cost_sweep.loc[
            (cost_sweep["policy"] == "dqn")
            & (cost_sweep["split"] == "test")
            & (cost_sweep["asset_group"] == "equity_index")
        ].sort_values("cost_rate_bp")
        self.assertTrue(dqn_equity["avg_cost_per_contract"].is_monotonic_increasing)

    def test_write_artifacts_outputs_expected_csvs(self) -> None:
        evaluator = ZhangEvaluator(
            trade_logs=make_trade_logs(),
            instruments=make_instruments(),
            config=ZhangEvalConfig(
                target_vol=0.05,
                cost_grid_bp=(1.0, 10.0),
                split="test",
                vol_span=2,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = evaluator.write_artifacts(tmp_dir)
            self.assertTrue(artifacts.contract_daily_returns_path.exists())
            self.assertTrue(artifacts.contract_metrics_path.exists())
            self.assertTrue(artifacts.daily_portfolios_unscaled_path.exists())
            self.assertTrue(artifacts.daily_portfolios_scaled_path.exists())
            self.assertTrue(artifacts.summary_unscaled_path.exists())
            self.assertTrue(artifacts.summary_scaled_path.exists())
            self.assertTrue(artifacts.cost_sweep_path.exists())
            self.assertTrue(artifacts.instrument_groups_path.exists())


if __name__ == "__main__":
    unittest.main()
