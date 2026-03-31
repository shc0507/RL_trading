# RL_trading

Week-1 implementation for the DRL trading project: public-data pipeline, Zhang-style daily features, a close-to-close simulated trading environment, baseline policies, and smoke-test backtests.

## What is implemented

- `rl_trading.data.PublicDailySource`: immediate daily-bar downloader using Stooq for `SPY`, `QQQ`, `IWM`, `DIA`, `EFA`, and `EEM`
- `rl_trading.data.InstitutionalCsvSource`: CSV adapter for Bloomberg, Refinitiv, WRDS, or manual exports
- `rl_trading.FeatureBuilder`: 60-step observations with normalized close, volatility-normalized returns over `21/42/63/252` days, MACD, RSI(30), and leakage QA
- `rl_trading.TradingEnv`: single-symbol environment with discrete or continuous target positions, transaction costs, raw PnL reward, and Zhang-style volatility-scaled reward
- `rl_trading.Backtester`: baseline evaluation with portfolio metrics and per-symbol reports
- `rl_trading.smoke`: end-to-end smoke runner

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Run the smoke test

```bash
uv run python -m rl_trading.smoke
```

Artifacts are written under `artifacts/`.
