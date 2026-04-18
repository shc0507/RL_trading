You are doing a code review of four files in a PyTorch project that recreates Zhang, Zohren, Roberts (2020) "Deep Reinforcement Learning for Trading" (JFDS). Do NOT edit files — review only.

**Files to review (read in full):**
- /Users/haochen/projects/RL_trading/rl_trading/trainer.py
- /Users/haochen/projects/RL_trading/rl_trading/baselines.py
- /Users/haochen/projects/RL_trading/rl_trading/metrics.py
- /Users/haochen/projects/RL_trading/rl_trading/evaluate.py

**Supporting files (read only as needed):**
- /Users/haochen/projects/RL_trading/rl_trading/env.py (reward formula — baselines reimplement it)
- /Users/haochen/projects/RL_trading/rl_trading/agents.py (to check trainer's per-agent dispatch)
- /Users/haochen/projects/RL_trading/rl_trading/config.py (defaults)

**Read the paper directly to avoid relying on paraphrases.** The PDF is at:
/Users/haochen/projects/RL_trading/docs/Deep-Reinforcement-Learning-for-Trading (1).pdf

Focus on pages 6–8. Pages 6–7 describe the training scheme: per-asset-class models (not per-ticker), retraining every 5 years, 10% of training data as CV set, early stopping with patience 20, dropout. Page 7 lists baseline formulas: Long, Sign(r_{t-252}), MACD signal φ(·). Page 8 lists metrics (E(R), std, DD, Sharpe, Sortino, MDD, Calmar, %+ve, AvgP/AvgL) and mentions portfolio-level volatility targeting. Use `pages: "6-8"`.

**What I want you to check (think independently — flag anything else that looks wrong):**

1. **trainer.py**:
   - Early stopping: uses validation Sharpe with patience=20. Paper says 20 epochs patience — consistent.
   - Per-episode flow: DQN stores + trains per step, PG stores per step then `train_episode` at end, A2C stores + trains when buffer full.
   - End-of-episode A2C flush temporarily sets `batch_size = 1` and trains once. Is this sensible? Could cause a noisy final update. Flag.
   - PG "train_episode" is called after the loop, but the loop only exits when `done` is True, which happens after the terminal transition — is the last reward included? Trace the indexing.
   - The isinstance dispatch pattern is fine but brittle. Not a bug, just a nit.
   - `_compute_sharpe` uses `arr.mean() / arr.std() * √252`. This is the Sharpe of daily *rewards* (which are vol-scaled trade returns). Is it meaningful? Compare to metrics.py.
   - Training prints every epoch — OK.
   - Deviation from paper: paper trains one model per asset class (mentioned in the TODO); trainer trains per-symbol. Don't re-flag the deviation itself but note any code assumption that hardcodes single-symbol.

2. **baselines.py**:
   - `sign_r` uses `ret_252` (raw 252-day return). Paper's Sign(R) baseline uses sign(r_{t-252,t}). Correct.
   - `macd_signal` uses `feat["macd_signal"]` (the combined φ-transformed signal, already clipped conceptually). Code clips to [-1, 1]. Is that needed? φ(x) is bounded roughly in [-0.5, 0.5] already.
   - `compute_baseline_rewards` reimplements env's reward formula. Check it matches env.py exactly — especially the `position_return = vol_scale * pos * daily_ret` and TC formula. Any drift between baseline and env reward math?
   - The `n = min(len(positions), len(prices) - 1)` guard: is `positions` always the right length?

3. **metrics.py**:
   - `compute_metrics` formulas vs paper Exhibit 2:
     - E(R) = annualized mean of daily returns (×252). ✓
     - Std(R) = annualized daily std (×√252). ✓
     - DD = annualized downside deviation (only r<0, std × √252). ✓
     - Sharpe = E(R)/Std(R). ✓
     - Sortino = E(R)/DD. ✓
     - MDD = max peak-to-trough in cumulative returns. ✓
     - Calmar = E(R) / MDD. ✓
     - %+Ret = fraction positive days (returned as percentage 0–100, but paper's table shows fractions 0–1!). Check whether this mismatches the paper's units — evaluate.py builds a CSV that might mix %+Ret in % with others as fractions.
     - AvgP/AvgL = avg positive / |avg negative|. ✓
   - MDD uses cumulative sum not cumulative product. For small daily returns the approximation is OK, but flag that it differs from log/cumprod conventions.
   - `std(ddof=1)` for annualized std — standard.
   - Empty / short array path returns zeros. OK.

4. **evaluate.py**:
   - `_detect_device`: cuda > mps > cpu. Fine.
   - `_make_agent_factories` uses `n_features=STATE_DIM`. Verify that env outputs matching shape (should: 10 features + 1 position).
   - `_collect_rl_rewards`: runs a trained agent on test split with `training=False`. Builds `pd.Series` indexed by dates from `env.history["date"]`. Is that the right length (one per step)?
   - Portfolio construction: per method, concat per-symbol rewards by outer-join on dates, then `mean(axis=1).dropna()`. The `.dropna()` drops dates where ANY of the current symbols has NaN. Is that right, or should it use `.mean(skipna=True)` to partial-fill? Subtle bug potential: if one symbol starts later than others, outer join leaves NaN, and mean would be computed over available symbols — but `.dropna()` on the final 1-D series drops rows where the mean itself is NaN, which only happens if ALL symbols are NaN on that day. Actually: `mean(axis=1)` is NaN-safe in pandas by default (skipna=True), so `.dropna()` drops full-NaN rows. That's probably correct. Double-check.
   - `_portfolio_vol_scale`: uses 60-day expanding std shifted by 1. First 60 days not scaled (scale=1). Paper applies "portfolio-level volatility targeting" per Exhibit 2. Is this implementation defensible? Flag if you see issues (look-ahead if shift missing, NaN handling, the `< 60` fast path returning unscaled).
   - Metrics table format: `float_format="%.4f"` on CSV, print uses 10.4f. `%+Ret` is a percent — if it's printed as 52.34 but other columns are 0.02-type fractions, column alignment still works but readers might misread units.
   - Plot: single column of stacked subplots, one per group ("All" + per-class). Cumulative returns via `cumsum(r)`. OK.
   - Failure handling: try/except around each symbol × method, stores empty Series on failure. This swallows exceptions silently — for a research experiment, failures should surface. Flag.
   - UNIVERSE subset logic: if user passes `--symbols FOO` and FOO not in UNIVERSE, code adds `{"symbol": "FOO", "asset_class": "other"}`. That creates a new group "other" in the portfolio aggregation. Is that intended?
   - Data fetch range hardcoded: `start="2004-01-01", end="2025-12-31"`. These overlap config splits (TEST_END=2025-12-31). OK.

5. Any dead code, wrong docstring, misleading variable name, exception swallowing, or place where the training/eval pipeline could silently produce wrong numbers without raising.

**Report format (under ~500 words):**
- One short paragraph with your overall take.
- A bulleted list of findings. For each: `file.py:line` reference, one-line problem statement, severity (bug / spec-deviation / nit).
- If you're unsure about something, say "UNSURE:" and explain.

Report back to me — the main Claude instance will consolidate findings across categories.
