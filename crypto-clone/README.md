# Quant Trading Crypto Clone

This project is a separate clone of the original workspace at `/Users/corr/Documents/New project`.

The goal of this clone is to be a practical, strategy-first quant trading project for crypto:

- research ideas on historical data
- backtest them with a consistent feature set
- paper trade them with realistic portfolio constraints
- connect the same runtime to a real broker later

It is intentionally **not RL-first**.

RL is still kept in the repo for future experiments, but the main architecture now assumes this workflow:

1. build and test simple systematic strategies first
2. verify the signal with backtests and paper trading
3. add live execution only after the basic process is stable
4. compare RL later if it truly earns its place

## Why This Architecture

For a first real trading application, the biggest risks are usually not model sophistication. They are:

- bad or inconsistent data
- unrealistic assumptions about fills and costs
- poor position sizing
- weak guardrails around cash usage and order sizes
- no clean path from backtest to paper to live

Because of that, this project is organized around stable trading-system layers instead of around a specific model family.

The core stack is:

- `data`
- `features`
- `strategy`
- `portfolio sizing`
- `broker/execution`
- `backtest`
- `paper/live runtime`

That gives you a cleaner path to real deployment than starting with RL.

## Recommended API Stack

For a small crypto account, this clone currently assumes:

- historical research data: Yahoo Finance
- live execution: Kraken spot REST

Why this is a good first combination:

- Yahoo Finance is fast to prototype with and is fine for daily-bar research.
- Kraken is relatively straightforward to integrate without a heavy SDK.
- The code separates data and execution, so you can replace either side later.

If you later want to trade stocks instead, the same architecture can be reused with:

- historical market data from a stock data vendor
- Alpaca as the paper/live broker

## Current Project Shape

The most important files are:

- `trading/data/sources.py`
- `trading/data/pipeline.py`
- `trading/features.py`
- `trading/strategies.py`
- `trading/portfolio.py`
- `trading/backtest.py`
- `trading/live.py`
- `trading/brokers/paper.py`
- `trading/brokers/kraken.py`
- `trading/research/pairs_trading/pairs.py`
- `trading/research/pairs_trading/README.md`
- `configs/crypto_paper.json`

For strategy-specific research modules, the project is starting to use self-contained folders.
The first example is:

- `trading/research/pairs_trading/`

## Architecture Overview

### 1. Data Layer

`trading.data.PublicDailySource`

Responsibilities:

- download daily bars
- normalize symbols and metadata
- produce a canonical bar frame

Right now this is the fastest way to do research on:

- `BTC-USD`
- `ETH-USD`
- `SOL-USD`

The project keeps this isolated so you can later swap in:

- exchange-native history
- paid institutional data
- intraday or websocket feeds

### 2. Feature Layer

`trading.FeatureBuilder`

Responsibilities:

- compute normalized price features
- compute volatility-normalized returns
- compute MACD features
- compute RSI
- enforce warmup logic
- check for leakage

This layer is shared by both:

- backtesting
- live strategy evaluation

That is important because it keeps the research and execution paths aligned.

### 3. Strategy Layer

`trading/strategies.py`

Responsibilities:

- define a common strategy interface
- emit a signal in `[-1, 1]`
- stay independent from broker/order mechanics

Built-in strategies:

- `LongOnlyStrategy`
- `Sign12MStrategy`
- `MACDStrategy`
- `RSIMeanReversionStrategy`

The important design choice here is:

- strategies produce conviction
- portfolio code decides sizing

That separation makes it much easier to compare strategies honestly.

### 4. Portfolio Layer

`trading/portfolio.py`

Responsibilities:

- apply position-size limits
- preserve a cash buffer
- enforce minimum order notional
- convert target exposure changes into actual buy/sell orders

This is one of the most useful layers in the project because it is where a lot of live-trading realism begins.

### 5. Broker Layer

`trading/brokers`

Responsibilities:

- expose a broker-agnostic interface
- allow the same runtime to work in paper or live mode

Implemented brokers:

- `PaperBroker`
- `KrakenBroker`

`PaperBroker` is for:

- dry runs
- local testing
- paper-trading style simulation

`KrakenBroker` is for:

- real spot execution
- account balance lookup
- market order submission

### 6. Backtest Layer

`trading/backtest.py`

Responsibilities:

- run strategies through the environment
- aggregate symbol-level returns
- compute portfolio metrics

This lets you keep using the existing research engine while the project becomes more production-oriented.

### 7. Live Runtime

`trading/live.py`

Responsibilities:

- fetch recent bars
- compute fresh features
- score the selected strategy
- run sizing logic
- send paper or real orders
- persist a JSON cycle report

That file is the bridge between research and execution.

## Folder Guide

```text
crypto-clone/
├── configs/
│   └── crypto_paper.json
├── trading/
│   ├── brokers/
│   │   ├── base.py
│   │   ├── kraken.py
│   │   └── paper.py
│   ├── data/
│   │   ├── pipeline.py
│   │   └── sources.py
│   ├── backtest.py
│   ├── config.py
│   ├── env.py
│   ├── features.py
│   ├── live.py
│   ├── policies.py
│   ├── portfolio.py
│   ├── rl_train.py
│   ├── smoke.py
│   └── strategies.py
└── tests/
```

## What `policies.py` Means Now

`trading/policies.py` still exists only for backward compatibility.

The project now treats:

- `strategies.py` as the real home for trading logic
- `policies.py` as an alias layer for older code, including the RL experiments

So if you add a new trading idea, prefer adding it to `strategies.py`.

## What RL Means Now

RL is now an optional sidecar, not the center of the project.

That means:

- the main project should work even if you never use RL
- new production logic should target the strategy and execution interfaces
- RL can be used later as a comparative research experiment

A healthy way to think about RL here is:

- baseline strategies first
- prove that the whole stack works
- then ask whether RL adds value beyond simpler methods

## Setup

This clone currently relies on the Python environment from the original workspace:

- Python: `/Users/corr/Documents/New project/.venv/bin/python`

If `uv` is not on your shell path, you can still run everything with that interpreter directly.

## Run The Test Suite

```bash
'/Users/corr/Documents/New project/.venv/bin/python' -m unittest discover -s tests -v
```

## Smoke Backtests

### Equities

```bash
'/Users/corr/Documents/New project/.venv/bin/python' -m trading.smoke --market equity
```

### Crypto

```bash
'/Users/corr/Documents/New project/.venv/bin/python' -m trading.smoke --market crypto --start-date 2017-01-01 --symbols BTC-USD,ETH-USD,SOL-USD
```

Artifacts are written under `artifacts/`.

The smoke run currently compares:

- `long_only`
- `sign_12m`
- `macd`
- `rsi_mean_reversion`

## Live / Paper Runtime

### Config File

See:

- `configs/crypto_paper.json`

Current defaults:

- market: `crypto`
- symbols: `BTC-USD`, `ETH-USD`, `SOL-USD`
- max symbol weight: `30%`
- cash buffer: `10%`
- min order notional: `$25`
- initial paper capital: `$2000`

### Run One Paper-Trading Cycle

```bash
'/Users/corr/Documents/New project/.venv/bin/python' -m trading.live --config configs/crypto_paper.json --strategy macd --broker paper --once --execute
```

That writes:

- `artifacts/live/latest_cycle.json`
- an archived cycle file under `artifacts/live/`

### Run A Dry Run Against Kraken

```bash
export KRAKEN_API_KEY=...
export KRAKEN_API_SECRET=...
'/Users/corr/Documents/New project/.venv/bin/python' -m trading.live --config configs/crypto_paper.json --strategy macd --broker kraken --once
```

Because `--execute` is not provided in that command, it will:

- compute the strategy
- size the trades
- log the cycle
- not submit live orders

### Run Live Execution

```bash
export KRAKEN_API_KEY=...
export KRAKEN_API_SECRET=...
'/Users/corr/Documents/New project/.venv/bin/python' -m trading.live --config configs/crypto_paper.json --strategy macd --broker kraken --once --execute
```

Use this only after you are comfortable with:

- the strategy logic
- sizing limits
- expected notional per trade
- your Kraken balances

## How A Live Cycle Works

One cycle of `trading.live` does this:

1. fetch recent bars for the configured symbols
2. build features
3. take the latest feature-ready row for each symbol
4. ask the broker for account state and current positions
5. ask the strategy for a signal per symbol
6. send that signal into the allocator
7. convert the result into a buy/sell/no-trade decision
8. optionally submit orders
9. write a JSON report

This is the exact flow you want in an early real-trading system because each step is easy to inspect.

## How To Add A New Strategy

Add a class in `trading/strategies.py` that subclasses `BaseStrategy`.

Implement:

- `signal(self, observation) -> float`

Guidelines:

- return values in `[-1, 1]`
- keep the logic stateless at first
- keep sizing decisions out of the strategy
- avoid broker-specific assumptions inside the strategy

Then register it in:

- `STRATEGY_REGISTRY`

After that, the same strategy can be used in:

- backtests
- paper trading
- live trading

## How To Replace Yahoo Finance Later

When you outgrow Yahoo Finance, the cleanest next step is:

1. add a new bar source in `trading/data/sources.py`
2. keep the canonical output schema the same
3. reuse `FeatureBuilder`, `Backtester`, `LiveTrader`, and the brokers

That is exactly why the project keeps the source adapter separate from the rest of the pipeline.

## Practical Development Roadmap

If you want to turn this into a serious small-account system, I would do the next steps in this order:

1. Add one more simple strategy you actually understand deeply.
2. Add slippage assumptions to the backtest layer.
3. Add a trade journal / reporting notebook or HTML report.
4. Add a daily portfolio snapshot artifact.
5. Add reconciliation between submitted orders and broker balances.
6. Upgrade market data quality.
7. Only then consider intraday or RL expansion.

## Risk Notes

This project is still a starter system, not a production-grade trading platform.

Missing pieces you would want before scaling capital:

- exchange-specific precision rules and minimum sizes
- stronger live fill reconciliation
- retry handling and idempotent client order IDs
- slippage-aware backtests
- better scheduling and health checks
- alerting
- persistent portfolio history

For a `$2000` account, that is fine for learning, but it should shape expectations.

## Test Coverage Added In This Clone

The clone includes tests for:

- data pipeline
- features
- environment
- backtester
- RL helper code
- paper broker
- live trader cycle
- target allocator
- strategy registry and baseline strategy behavior

## Summary

This clone is now meant to be your practical quant sandbox:

- simple enough to understand end to end
- structured enough to grow into a real application
- flexible enough to support non-RL and RL work

If you keep building from here, the right default mindset is:

- make the system robust first
- make the strategy better second
- make the model more complex only if it earns the complexity
