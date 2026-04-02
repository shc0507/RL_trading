# Pairs Trading

This folder contains the pair-trading research workflow for the project.

Main code:

- `pairs.py`

The purpose of this module is to answer a very specific research question:

1. Are two tickers statistically tied together strongly enough to trade as a pair?
2. If they are, what entry and exit thresholds look reasonable on historical data?
3. Do those thresholds still hold up on a held-out test period?

## Strategy Idea

Pairs trading is a relative-value strategy.

Instead of asking:

- "Will ticker A go up?"

it asks:

- "Has the relationship between ticker A and ticker B moved far enough away from normal that it may revert?"

The classic intuition is:

- two highly related assets often move together over time
- short-term dislocations can happen
- when the dislocation becomes unusually large, you bet on convergence rather than outright direction

That is why pairs trading is often described as:

- long one leg
- short the other leg
- profit from the spread mean-reverting

## What This Implementation Does

The code in `pairs.py` follows this workflow:

1. load and align two price series
2. fit a long-run hedge relationship on the training sample
3. test whether the residual spread looks stationary
4. if it does, compute rolling z-scores of the spread
5. tune entry and exit thresholds with expanding-window cross-validation
6. freeze the selected thresholds
7. run a final held-out backtest on the test period

This is intentionally simpler than a production stat-arb stack, but it is much better than:

- fitting on the full sample
- eyeballing thresholds
- testing on the same data used for tuning

## Cointegration In Plain English

Two price series can both wander over time and still maintain a stable long-run relationship.

That is the basic idea behind cointegration.

In this implementation:

- we regress `log(price_x)` on `log(price_y)`
- we compute the residual spread
- we test whether that spread itself looks mean-reverting

If the spread does not look stationary, the pair is rejected.

That matters because a high correlation alone is not enough.

Two assets can be highly correlated and still drift apart in a way that breaks a convergence trade.

## Why Use Log Prices

The code models:

- `log(price_x) = intercept + hedge_ratio * log(price_y) + spread`

Log prices make the relationship easier to interpret and often behave more cleanly in proportional terms.

This is a standard choice in many pairs-trading workflows.

## What The Hedge Ratio Means

The hedge ratio is the slope from the regression between the two log-price series.

You can think of it as the amount of ticker Y needed to hedge ticker X in the spread model.

The code uses that hedge ratio in two places:

- to define the spread
- to size the long/short legs in the pair return calculation

It is estimated on the training window only and then frozen during validation or test.

That is important for leakage control.

## Cointegration Test Used Here

This module uses a lightweight Engle-Granger-style approach:

1. regress one log price on the other
2. compute the residual spread
3. regress the change in spread on the lagged spread
4. use the t-stat on the lagged coefficient as a rough ADF-style stationarity filter

This is a practical research approximation.

It is useful because:

- it keeps dependencies light
- it makes the logic easy to inspect
- it is enough for an internal research pipeline

But it is not the last word statistically.

If you wanted to make this more rigorous later, you could switch to:

- `statsmodels` Engle-Granger tools
- Johansen tests
- rolling structural-break checks

## Trading Logic

Once the spread is normalized into a rolling z-score:

- if z-score is very negative, the spread is "too low"
- if z-score is very positive, the spread is "too high"

The current rules are:

- enter long spread when `z <= -entry_threshold`
- enter short spread when `z >= entry_threshold`
- exit when `abs(z) <= exit_threshold`

In this implementation:

- long spread means long `X` and short `Y`
- short spread means short `X` and long `Y`

The idea is always the same:

- bet that the spread moves back toward its normal range

## Why Threshold Tuning Exists

Pairs trading is very sensitive to thresholds.

If your entry threshold is too small:

- you overtrade
- pay too much in costs
- enter on weak deviations

If your entry threshold is too large:

- you get very few trades
- you can miss useful dislocations

The exit threshold matters too:

- exiting too early may cut profits
- exiting too late may give back convergence gains

So the module tunes:

- `entry_threshold`
- `exit_threshold`

instead of hard-coding one arbitrary choice.

## Why Use Expanding-Window CV

The cross-validation logic is designed to be time-aware.

It does not shuffle the data.

Instead it:

- trains on the past
- validates on the next unseen block
- expands the training window fold by fold

That is closer to how you would really retune in live trading.

It also helps reduce the risk of choosing thresholds that only worked because of accidental hindsight.

## Why Keep A Training Tail In Test

The backtest carries a short tail of the training sample into the evaluation window.

That is only so the first test observations can compute a rolling mean and rolling standard deviation for z-score normalization.

This does not leak future test information.

It just avoids starting the test window with undefined rolling features.

## Outputs

The module produces:

- `CointegrationResult`
- `ThresholdSelectionResult`
- `PairsBacktestReport`
- `PairsResearchResult`

These are designed to answer different questions:

- is the pair structurally tradeable?
- which thresholds looked best in CV?
- how did the final test period behave?

## Current Assumptions

This implementation is a good research starter, but it makes simplifying assumptions:

- daily bars
- one fixed hedge ratio per train/validation/test segment
- simple transaction cost model
- no borrow fees
- no short-sale constraints
- no rolling re-estimation inside the test window
- no regime-switching logic

Those are fine for a first internal research tool, but they matter if you want to put real money behind the strategy.

## Main Risks And Failure Modes

Pairs trading can fail badly when:

- the relationship between the assets structurally changes
- one leg experiences an event the other leg does not
- the spread stops mean-reverting
- costs eat the edge
- the model was fit on a regime that no longer exists

That is why passing a historical cointegration test is necessary, but not sufficient.

## Good Pairs To Start With

For equities, more natural pairs are usually things like:

- two very similar ETFs
- two share classes
- two companies with very similar economics

Examples:

- `SPY` / `IVV`
- `GOOG` / `GOOGL`
- `XLE` / `VDE`

For crypto, true pairs trading is trickier because relationships can be less stable than they look.

If you test crypto pairs, treat the results more cautiously.

## How To Run It

Example:

```bash
./.venv/bin/python -m trading.pairs \
  --symbol-x SPY \
  --symbol-y IVV \
  --start-date 2018-01-01 \
  --train-end-date 2023-12-31 \
  --test-end-date 2025-12-31 \
  --market equity \
  --entry-thresholds 1.0,1.5,2.0,2.5 \
  --exit-thresholds 0.0,0.5,1.0
```

The command will:

- test cointegration on the training period
- tune thresholds on training folds
- run a held-out test
- print the selected thresholds and test metrics

## How To Extend It

The next upgrades I would consider are:

1. rolling re-estimation of the hedge ratio
2. rolling re-check of cointegration
3. max holding period rules
4. stop-loss / stop-out logic
5. richer transaction cost and borrow modeling
6. structured artifact export for research notebooks

## Why This Folder Exists

This project now has multiple strategy ideas and they should not all live as one-off top-level files.

This folder structure makes it easier to keep each strategy self-contained:

- code
- explanation
- assumptions
- future extensions

That should make the project easier for you to navigate as more strategies get added.
