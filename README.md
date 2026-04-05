# RL_trading

Week-1 foundation for DRL trading experiments: market-data ingestion, Zhang-style daily features, a close-to-close simulated trading environment, baseline policy backtests, and an end-to-end smoke run.

The repository does not yet include RL agent training. Its current role is to provide the data, environment, and evaluation baseline that later RL work can build on.

## Current status

- Public daily data is fetched from Yahoo Finance for `SPY`, `QQQ`, `IWM`, `DIA`, `EFA`, and `EEM`
- Fetch retries, raw-source snapshots, fetch manifests, and bar-quality checks are implemented
- Feature generation, schema validation, and leakage checks are implemented
- A single-symbol trading environment and portfolio-level backtester are implemented
- Current benchmark policies are rule-based baselines: `long_only`, `sign_12m`, and `macd`
- End-to-end smoke artifacts are written under `artifacts/`

## What is implemented

- `rl_trading.data.PublicDailySource`: public daily-bar downloader backed by Yahoo Finance
- `rl_trading.data.InstitutionalCsvSource`: CSV adapter for Bloomberg, Refinitiv, WRDS, or manual exports
- `rl_trading.FeatureBuilder`: 60-step observations with normalized close, volatility-normalized returns over `21/42/63/252` days, MACD, RSI(30), and leakage QA
- `rl_trading.TradingEnv`: single-symbol environment with discrete or continuous target positions, transaction costs, raw PnL reward, and Zhang-style volatility-scaled reward
- `rl_trading.Backtester`: baseline evaluation with portfolio metrics, daily returns, trade logs, and per-symbol reports
- `rl_trading.smoke`: end-to-end smoke runner for the default ETF universe and baseline policies

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Run the smoke test

```bash
uv run python -m rl_trading.smoke
```

## Outputs

Running the smoke test writes artifacts under `artifacts/`, including:

- `artifacts/raw/`: per-source raw fetch snapshots
- `artifacts/processed/`: normalized bars, features, and instrument metadata
- `artifacts/manifests/`: fetch metadata and split definitions
- `artifacts/qa/`: bar-quality and leakage-check output
- `artifacts/smoke/`: summary metrics, daily returns, trade logs, and per-symbol metrics

## Data Quality Tradeoff

The default pipeline now checks that requested symbols share nearly the same observed trading calendar and that each symbol reaches the requested `end_date` within a small business-day lag.

This is intentional for the current default universe of broad US-listed ETFs. The tradeoff is that the same check can be too strict for mixed universes that legitimately trade on different calendars or have different holiday schedules. If you expand the universe that way, tune `MarketDataPipeline(..., min_symbol_calendar_coverage=..., max_symbol_end_lag_business_days=...)` instead of treating every coverage failure as bad data.
