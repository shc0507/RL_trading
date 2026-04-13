# RL_trading

Week-1 foundation for DRL trading experiments: market-data ingestion, Zhang-style daily features, a close-to-close simulated trading environment, baseline policy backtests, and an end-to-end smoke run.

The repository now also includes a first RL baseline family: DQN, Double DQN, Dueling DQN, LSTM DQN, A2C, PPO, TD3, and SAC on top of the existing trading stack. The current evaluation layer is Zhang-aligned in reporting on the repo's ETF dataset, but it is not a full reproduction of Zhang et al.'s futures-based experiments.

## Current status

- Public daily data is fetched from Yahoo Finance for `SPY`, `QQQ`, `IWM`, `DIA`, `EFA`, and `EEM`
- Fetch retries, raw-source snapshots, fetch manifests, and bar-quality checks are implemented
- Feature generation, schema validation, and leakage checks are implemented
- A single-symbol trading environment and portfolio-level backtester are implemented
- Current benchmark policies are rule-based baselines: `long_only`, `sign_12m`, and `macd`
- Repo-local DQN, Double DQN, Dueling DQN, and LSTM DQN variants are implemented
- Repo-local A2C, PPO, TD3, and SAC trainers are implemented
- End-to-end smoke artifacts are written under `artifacts/`

## What is implemented

- `rl_trading.data.PublicDailySource`: public daily-bar downloader backed by Yahoo Finance
- `rl_trading.data.InstitutionalCsvSource`: CSV adapter for Bloomberg, Refinitiv, WRDS, or manual exports
- `rl_trading.FeatureBuilder`: 60-step observations with normalized close, volatility-normalized returns over `21/42/63/252` days, MACD, RSI(30), and leakage QA
- `rl_trading.TradingEnv`: single-symbol environment with discrete or continuous target positions, transaction costs, raw PnL reward, and Zhang-style volatility-scaled reward
- `rl_trading.Backtester`: baseline evaluation with portfolio metrics, daily returns, trade logs, and per-symbol reports
- `rl_trading.smoke`: end-to-end smoke runner for the default ETF universe and baseline policies
- `rl_trading.train_dqn`: DQN-family training entrypoint that trains on `train`, selects checkpoints on `val`, evaluates on `train/val/test`, and writes Zhang-aligned evaluation artifacts
- `rl_trading.train_on_policy`: A2C/PPO training entrypoint for discrete trading actions
- `rl_trading.train_continuous`: TD3/SAC training entrypoint for continuous target-position actions
- `rl_trading.rl.policy_gradient`: A2C/PPO actor-critic modules
- `rl_trading.rl.continuous_control`: TD3/SAC modules
- `rl_trading.plot_rl`: plotting helper for both legacy repo charts and Zhang-aligned evaluation figures

## Reuse notes

The adapted homework-to-repo RL notes live in `docs/rl_reuse_notes.md`.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Run the smoke test

```bash
uv run python -m rl_trading.smoke
```

## Run the DQN baseline

```bash
uv run python -m rl_trading.train_dqn
```

To run a Zhang-closer recurrent Double Dueling DQN variant:

```bash
uv run python -m rl_trading.train_dqn --network-type lstm --double-dqn --dueling --policy-name lstm_double_dueling_dqn
```

## Run A2C or PPO

```bash
uv run python -m rl_trading.train_on_policy --algorithm ppo
```

```bash
uv run python -m rl_trading.train_on_policy --algorithm a2c
```

## Run TD3 or SAC

```bash
uv run python -m rl_trading.train_continuous --algorithm sac
```

```bash
uv run python -m rl_trading.train_continuous --algorithm td3
```

## Render RL plots

```bash
uv run python -m rl_trading.plot_rl --artifact-dir artifacts/<run>/rl --split test
```

The default plot mode is Zhang-aligned and reads or refreshes `artifacts/<run>/rl/zhang_eval/`. To render the older internal comparison charts instead:

```bash
uv run python -m rl_trading.plot_rl --artifact-dir artifacts/<run>/rl --split test --style legacy
```

## Outputs

Running the smoke test writes artifacts under `artifacts/`, including:

- `artifacts/raw/`: per-source raw fetch snapshots
- `artifacts/processed/`: normalized bars, features, and instrument metadata
- `artifacts/manifests/`: fetch metadata and split definitions
- `artifacts/qa/`: bar-quality and leakage-check output
- `artifacts/smoke/`: summary metrics, daily returns, trade logs, and per-symbol metrics
- `artifacts/rl/`: RL checkpoints, training curve, summary metrics, legacy evaluation reports, and Zhang-aligned evaluation artifacts
- `artifacts/rl/zhang_eval/`: scaled and unscaled portfolio summaries, contract metrics, daily portfolio return series, cost sweeps, and instrument grouping metadata
- `artifacts/rl/plots/`: legacy and Zhang-aligned figures

## Zhang Alignment Scope

The repo now distinguishes between:

- the current training baseline: plain DQN on the default ETF universe
- the current reporting layer: Zhang-aligned evaluation and plotting conventions on that ETF dataset

That means the figures and summary tables now follow the paper more closely in additive cumulative trade returns, portfolio-level volatility scaling, asset-group reporting, and transaction-cost sweeps. It still does not claim the repo reproduces the paper's model architecture, futures dataset, or rolling retraining protocol.

## Data Quality Tradeoff

The default pipeline now checks that requested symbols share nearly the same observed trading calendar and that each symbol reaches the requested `end_date` within a small business-day lag.

This is intentional for the current default universe of broad US-listed ETFs. The tradeoff is that the same check can be too strict for mixed universes that legitimately trade on different calendars or have different holiday schedules. If you expand the universe that way, tune `MarketDataPipeline(..., min_symbol_calendar_coverage=..., max_symbol_end_lag_business_days=...)` instead of treating every coverage failure as bad data.
