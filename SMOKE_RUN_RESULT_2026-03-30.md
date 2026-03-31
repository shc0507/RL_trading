# Smoke Run Result

This note summarizes the successful end-to-end smoke run we executed after switching the public daily data source to Yahoo Finance.

Command used:

```bash
.venv/bin/python -m rl_trading.smoke
```

Artifacts were written under:

- `artifacts/processed/`
- `artifacts/manifests/`
- `artifacts/qa/`
- `artifacts/smoke/`

## What the smoke run did

1. Downloaded daily OHLCV market data for the default ETF set
2. Built the feature table
3. Validated the feature frame
4. Ran leakage checks
5. Backtested three baseline policies across train / val / test
6. Saved CSV outputs for daily returns, trades, and symbol metrics
7. Printed a summary table

## Policies tested

- `long_only`
- `macd`
- `sign_12m`

## Headline results

### Best overall baseline

`long_only` was the strongest baseline in all three splits.

Why:

- highest annualized return across train / val / test
- strongest Sharpe across the splits
- almost no trading turnover after initial positioning
- very low total transaction cost compared with the active strategies

### Strategy behavior summary

`long_only`
- strongest overall returns
- lowest turnover
- lowest transaction cost
- acts like a buy-and-hold baseline

`sign_12m`
- positive in all three splits
- weaker than `long_only`
- noticeably higher turnover and higher cost
- reasonable simple trend-following baseline

`macd`
- negative in train
- positive in val
- slightly negative in test
- highest turnover and highest transaction costs
- appears to overtrade in this setup

## Smoke summary table

| Policy | Split | Annualized Return | Annualized Volatility | Sharpe | Hit Rate | Avg Win | Avg Loss | Num Days | Avg Daily Turnover | Total Turnover | Total Transaction Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_only | test | 1.431955 | 2.178162 | 0.657414 | 0.548607 | 0.097773 | -0.106242 | 1759.0 | 0.000569 | 1.000000 | 1258.259166 |
| macd | test | -0.032682 | 1.495081 | -0.021860 | 0.514497 | 0.062236 | -0.066220 | 1759.0 | 0.046675 | 82.101896 | 3310.209640 |
| sign_12m | test | 0.230027 | 1.829574 | 0.125727 | 0.527572 | 0.078837 | -0.086107 | 1759.0 | 0.045859 | 80.666667 | 3029.774660 |
| long_only | train | 0.954998 | 2.292233 | 0.416624 | 0.545004 | 0.101986 | -0.113832 | 2422.0 | 0.000413 | 1.000000 | 558.238538 |
| macd | train | -0.043581 | 1.543128 | -0.028242 | 0.526012 | 0.063051 | -0.070336 | 2422.0 | 0.048671 | 117.880106 | 1572.881944 |
| sign_12m | train | 0.204414 | 1.991102 | 0.102664 | 0.530966 | 0.082999 | -0.092229 | 2422.0 | 0.042389 | 102.666667 | 1387.211667 |
| long_only | val | 2.073444 | 2.256930 | 0.918701 | 0.560425 | 0.098009 | -0.106236 | 753.0 | 0.001328 | 1.000000 | 465.435560 |
| macd | val | 0.668218 | 1.430060 | 0.467266 | 0.527224 | 0.059926 | -0.061219 | 753.0 | 0.046792 | 35.234258 | 1014.458881 |
| sign_12m | val | 1.281666 | 1.942804 | 0.659699 | 0.551129 | 0.081560 | -0.088810 | 753.0 | 0.042497 | 32.000000 | 1081.155705 |

## How to read the key columns

`annualized_return`
- approximate yearly return implied by the return series

`annualized_volatility`
- yearly volatility implied by the return series

`sharpe`
- return divided by volatility
- higher is better

`hit_rate`
- fraction of days with positive return

`avg_win`
- average return on winning days

`avg_loss`
- average return on losing days

`num_days`
- number of daily observations in that split

`avg_daily_turnover`
- average daily change in position
- higher means more trading

`total_turnover`
- total cumulative position changes over the full split

`total_transaction_cost`
- total cost paid because of trading

## Plain-English takeaway

For this repo's current setup and ETF universe:

- `long_only` is the strongest and cheapest baseline
- `sign_12m` is a reasonable active baseline
- `macd` appears too expensive / too active for the edge it generates here

This gives a useful baseline target for future RL agents:

- an RL strategy should ideally beat `sign_12m`
- and eventually justify itself against `long_only`, especially after costs
