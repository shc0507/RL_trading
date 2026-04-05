# Study Summary: Python and the RL Trading Project

This note summarizes what you learned today from our walkthrough of the repo and the Python concepts behind it.

## 1. Big Picture of the Project

The project is building the foundation for a reinforcement learning trading system.

High-level flow:

1. Get market data
2. Clean and standardize the data
3. Build features from the price data
4. Create a trading environment
5. Run simple baseline policies
6. Backtest them
7. Compute performance metrics

Main files and roles:

- `rl_trading/smoke.py`: End-to-end smoke test that runs the whole pipeline
- `rl_trading/data/pipeline.py`: Orchestrates data processing and saving outputs
- `rl_trading/data/sources.py`: Reads market data from public web data or local CSVs
- `rl_trading/features.py`: Converts price data into features
- `rl_trading/env.py`: Simulated trading environment
- `rl_trading/policies.py`: Baseline strategies that output actions
- `rl_trading/backtest.py`: Runs policies through the environment and collects results
- `rl_trading/metrics.py`: Computes performance statistics from returns

## 2. What `smoke.py` Does

`smoke.py` is an end-to-end test of the whole system.

It:

1. Builds the data pipeline
2. Fetches market data
3. Builds features
4. Creates the trading environment and backtester
5. Runs simple baseline policies
6. Saves detailed outputs
7. Saves a final summary CSV

The idea of a "smoke test" is not "final training." It is a quick way to confirm the whole system can run from start to finish.

## 3. Python Command-Line Basics

### `python -m rl_trading.smoke`

The `-m` means:

"Run this Python module as a program."

So:

- `rl_trading` is the package
- `smoke` is the module (`smoke.py`)

This is useful because the file uses package-relative imports like:

```python
from .backtest import Backtester
```

### `main()`

The `main()` function in `smoke.py`:

1. Creates a command-line parser
2. Reads `--output-dir`
3. Calls `run_smoke(...)`
4. Reads the generated summary CSV
5. Prints it to the terminal

### `parser`

`parser` is a command-line argument parser.

Its job:

- define what arguments the program accepts
- read what the user typed
- convert that into Python values

Example:

```python
parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
args = parser.parse_args()
```

This means:

- the script accepts `--output-dir`
- if not provided, it uses the default value

## 4. Class vs Object vs Method

You learned this distinction:

- `class`: the blueprint
- `object` (instance): the actual thing created from the blueprint
- `method`: a function defined inside the class

Example from the repo:

```python
pipeline = MarketDataPipeline(output_dir=output_path)
pipeline.build(...)
```

Meaning:

- `MarketDataPipeline` is the class
- `pipeline` is the object
- `build(...)` is the method

## 5. `self`

`self` means:

"the current object itself"

You use `self` inside class methods when the method needs access to the object's own data.

Example:

```python
self.output_dir
self.feature_columns
self.position
```

Normal functions do not use `self`.

## 6. `@property`

`@property` lets you access a method like an attribute.

Example:

```python
@property
def feature_columns(self):
    return DEFAULT_FEATURE_COLUMNS
```

You use it like this:

```python
builder.feature_columns
```

not:

```python
builder.feature_columns()
```

It looks like a field, but runs a function behind the scenes.

## 7. `@dataclass`

`@dataclass` helps define classes whose main purpose is to hold structured data.

Example:

```python
@dataclass(slots=True)
class BuildArtifacts:
    bars: pd.DataFrame
    features: pd.DataFrame
    bars_path: Path
```

What you learned:

- it reduces boilerplate
- it can still include methods
- it is not limited to "data only" in a strict sense

### `slots=True`

This makes the class more fixed and a bit stricter/lightweight.

You can think of it as:

"This object has a predefined set of storage slots."

## 8. Type Hints You Learned

### `name: str`

Means:

"`name` should be a string."

### `output_dir: str | Path = DEFAULT_OUTPUT_DIR`

Means:

- this parameter can be a string or a `Path`
- if no value is passed, use `DEFAULT_OUTPUT_DIR`

### `splits: dict[str, tuple[str, str]] | None = None`

Means:

- `splits` can be a dictionary
- keys are strings, like `"train"`
- values are 2-item tuples of strings, like `("2005-01-01", "2015-12-31")`
- or it can be `None`
- default is `None`

## 9. Decorators and `@`

`@` means a decorator is being applied.

Rough mental model:

```python
@dataclass
class A:
    ...
```

is similar in spirit to:

```python
class A:
    ...

A = dataclass(A)
```

So the class or function is created, then "wrapped" or processed.

## 10. Python Dictionary Patterns

### `metrics = {...}`

That is a dictionary.

It maps names to values:

```python
metrics = {
    "annualized_return": annual_return,
    "sharpe": float(sharpe),
}
```

### `row.get("ret_252_raw", 0.0)`

This means:

"Get the value for `ret_252_raw`; if it does not exist, use `0.0`."

### `float(row.get("ret_252_raw", 0.0) or 0.0)`

This is a defensive pattern:

1. Try to get `ret_252_raw`
2. If missing, use `0.0`
3. If the result is still a falsy value like `None`, use `0.0`
4. Convert to float

You noticed correctly that this is somewhat redundant. That was a good observation.

### `**some_dict`

This expands a dictionary.

Example:

```python
{"policy": "macd", **report.portfolio_metrics}
```

This inserts the key-value pairs from `report.portfolio_metrics` directly into the new dictionary.

## 11. pandas / DataFrame / Series Basics

This was a major part of what you learned.

### DataFrame vs Series

- `DataFrame`: a table
- `Series`: usually one column or one row

Examples:

```python
frame["adj_close"]
```

returns a Series.

```python
row = frame.iloc[5]
```

returns one row, usually as a Series.

### Common pandas operations you learned

#### `frame["col"]`

Take one column.

#### `frame["new_col"] = ...`

Create or overwrite a column.

#### `sort_values("date")`

Sort rows by a column.

#### `reset_index(drop=True)`

Reset row numbering.

#### `iloc[...]`

Select rows/columns by integer position.

Example:

```python
row = self.episode_frame.iloc[self.pointer]
```

#### `loc[...]`

Select rows/columns using labels or conditions.

Example:

```python
frame.loc[frame["symbol"] == symbol]
```

#### `groupby(...)`

Split data into groups for separate processing.

Example:

```python
bar_frame.groupby("symbol")
```

#### `pd.concat(...)`

Combine multiple DataFrames into one.

#### `fillna(0.0)`

Replace missing values (`NaN`) with `0.0`.

#### `replace(0.0, np.nan)`

Replace a specific value with another value.

#### `isna()` / `notna()`

Check whether values are missing.

#### `any()` / `all()`

Test whether any or all values satisfy a condition.

### Time series operations you learned

#### `diff()`

Current value minus previous value.

#### `pct_change()`

Percent change from previous value.

#### `rolling(window).mean()` / `.std()`

Rolling window calculations.

Think:

"At each row, look back over the last N rows."

#### `ewm(...).mean()` / `.std()`

Exponentially weighted calculations.

Think:

"Use a weighted average where more recent data matters more."

#### `clip(lower=..., upper=...)`

Clamp values into a range.

Used in RSI and for final bounds.

#### `mask(condition, value)`

Replace values where a condition is true.

#### `where(condition, other)`

Keep values where a condition is true; replace the rest.

### `row.to_dict()`

This converts one pandas row into a normal Python dictionary.

Useful for passing a complete row to a policy in an easy-to-read format.

## 12. `pipeline.py`

This file is the data pipeline manager.

It does:

1. Prepare output directories
2. Fetch market data
3. Validate raw bars
4. Build features
5. Validate the feature table
6. Run leakage checks
7. Save CSV and JSON outputs
8. Validate split definitions
9. Return a `BuildArtifacts` object

### Directory creation loop

```python
for directory in (self.raw_dir, self.processed_dir, self.manifest_dir, self.qa_dir):
    directory.mkdir(parents=True, exist_ok=True)
```

Meaning:

"Create each of these directories if it does not already exist."

- `parents=True`: also create parent directories if needed
- `exist_ok=True`: do not error if the directory already exists

## 13. `sources.py`

This file defines where market data comes from.

Two main sources:

- `PublicDailySource`: download public daily data
- `InstitutionalCsvSource`: read local CSV files

### URL template and `{ticker}`

Example:

```python
"https://stooq.com/q/d/l/?i=d&s={ticker}"
```

This is a template string.

`{ticker}` is a placeholder that gets replaced later.

If you wanted literal braces, you would use:

```python
"{{ticker}}"
```

### `StringIO(text)` and `pd.read_csv(StringIO(text))`

If `text` is CSV content as a string, then:

- `StringIO(text)` turns it into a file-like object
- `pd.read_csv(...)` reads it as if it were a CSV file

### `column_aliases`

This is a translation map from many possible outside column names to the internal standard names.

Goal:

"Take messy real-world CSV columns and map them into the project's standard schema."

### `normalized`

`normalized` is the cleaned-up intermediate structure where the project builds a new set of standard columns from the raw CSV.

It is not directly "renaming in place." It is building a new normalized column mapping.

## 14. `features.py`

This file converts price data into model/environment features.

Important concepts:

- rolling windows
- volatility estimates
- normalized returns
- MACD
- RSI
- `window_ready`

### `window_ready`

This means:

"This row now has enough past data for all required features to be valid."

Rows before that are part of the warmup period.

### `check_leakage()`

Purpose:

"Check whether features accidentally use future data."

Idea:

1. Build full features using all data
2. Truncate the raw data at some point
3. Rebuild features only from past data
4. Compare the row
5. If the values match, there is no future leakage

### `validate_feature_frame()`

Purpose:

"Make sure the feature table is structurally valid."

Checks include:

- required columns exist
- no duplicate `(symbol, date)` rows
- dates are sorted within each symbol
- there are valid rows after warmup
- no NaNs remain in the ready area
- RSI stays within `[0, 100]`

## 15. `env.py`

This file turns the feature table into a step-by-step trading environment.

Key ideas:

- one episode trades one symbol
- `reset()` starts a new episode for one symbol and one split
- `step(action)` advances one day

### Important environment state variables

Examples:

```python
self.episode_frame: pd.DataFrame | None = None
self.symbol: str | None = None
self.split: str | None = None
```

These are placeholders that get filled after `reset(...)`.

### Observation

`_observation()` returns the current state:

- current symbol
- current split
- current date
- current position
- recent feature window
- current row features
- full current row as a dictionary

### Pointer

The pointer tracks the current row in the episode.

Important:

- `_observation()` does not advance time
- `step()` updates `self.pointer += 1`

### Turnover

```python
turnover = abs(action - self.position)
```

This measures how much the portfolio position changed.

It is used to model transaction cost.

### Raw return

```python
raw_return = action * pct_change - (cost_rate * turnover)
```

This is the strategy's step-level net return:

- market move times position
- minus trading cost

### Zhang return / Zhang reward

This uses volatility-scaled positions before computing return or reward.

Meaning:

"Evaluate the strategy under a risk-adjusted / volatility-targeted position scaling."

The metrics still mathematically work on Zhang returns, but their economic interpretation changes. They become more like performance under a risk-normalized return process rather than plain raw PnL.

## 16. `policies.py`

This file contains baseline policies that turn an observation into an action.

Examples:

- `LongOnlyPolicy`: always return `1.0`
- `Sign12MPolicy`: use the sign of 12-month raw return
- `MACDPolicy`: use `macd_signal` to produce an action

### Important insight

`LongOnlyPolicy` returning `1.0` does not mean "buy more forever."

It means:

"The target position is always full long exposure."

If the environment is already at position `1.0`, turnover is zero, so no new buying occurs.

## 17. `backtest.py`

This file runs a policy through the environment and collects evaluation results.

### `run(...)`

High-level flow:

1. Decide which symbols to run
2. Create empty log containers
3. Pick return/cost columns based on reward mode
4. For each symbol:
   - reset environment
   - repeatedly call `policy.act(...)`
   - call `env.step(action)`
   - save step logs
5. Compute metrics for each symbol
6. Aggregate across symbols into a portfolio daily return series
7. Compute overall portfolio metrics
8. Return an `EvalReport`

### Important insight

One episode in the environment trades one symbol.

But one backtest `run(...)` can loop over many symbols.

That is why both of these exist:

- `symbol_metrics`: per-symbol results
- `portfolio_metrics`: aggregate results across symbols

## 18. `metrics.py`

This file turns a return series into performance statistics.

It computes things like:

- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio
- hit rate
- average win / average loss
- turnover and cost metrics

### Important formulas you learned

#### Annualized return

Approximate:

```python
daily_mean_return * 252
```

#### Annualized volatility

Approximate:

```python
daily_std * sqrt(252)
```

#### Sharpe with protection

```python
sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
```

If volatility is zero, it returns `0.0` instead of dividing by zero.

### `metrics` object

The `metrics` variable is a dictionary.

It stores performance statistics as:

```python
{
    "annualized_return": ...,
    "sharpe": ...,
    ...
}
```

## 19. RL Concepts You Connected to the Repo

You realized this codebase is building the core of an RL trading setup.

Mapping:

- Features -> state / observation
- Policy output -> action
- Environment step result -> reward and next state
- Episode -> one symbol over one split
- Backtest -> evaluation loop

This repo is not yet the full RL trainer, but it already has:

- data layer
- feature layer
- environment layer
- baseline decision policies
- evaluation logic

## 20. Key Mental Models to Remember

### `env.py`

Think:

"one episode, one symbol, one split"

### `backtest.py`

Think:

"same policy, many symbols, aggregate everything"

### `raw_return`

Think:

"actual step-level net return under the chosen position"

### `zhang_return`

Think:

"risk-scaled step-level return"

### `metrics.py`

Think:

"take a return series and summarize how good or bad it was"

## 21. What You Got Better At Today

You made strong progress on:

- understanding Python classes, methods, and `self`
- understanding `@property`, `@dataclass`, and decorators
- reading type hints
- recognizing pandas DataFrame and Series operations
- understanding how time-series features are computed
- understanding what the RL trading environment is doing
- seeing the difference between one-symbol episodes and multi-symbol backtests
- understanding the difference between raw and risk-scaled returns

## 22. Good Questions You Asked

These were especially important questions:

- What is the difference between class, object, and method?
- Why do we use `self`?
- What does `@property` do?
- Why does `@dataclass` still allow methods?
- What does `**dict` mean?
- Why are there both `symbol_metrics` and `portfolio_metrics`?
- If one episode trades one symbol, why is there a portfolio?
- Does Zhang return still make the metrics meaningful?

Those are exactly the kinds of questions that help turn "reading code" into real understanding.

## 23. Suggested Next Steps

Good next topics for you:

1. Trace one full example with real numbers through `env.step()`
2. Add one new simple baseline policy yourself
3. Modify `smoke.py` to accept more CLI args like `--start-date`
4. Wrap `TradingEnv` into a gymnasium-compatible environment
5. Start plugging in a simple RL agent

