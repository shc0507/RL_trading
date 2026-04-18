# Subagent 4 — Training & Evaluation Review

**Scope.** `trainer.py`, `baselines.py`, `metrics.py`, `evaluate.py`
**Prompt.** `../review_prompts/4_training_and_evaluation.md`

> Annotations by the main reviewer are prefixed with **[MAIN]:**.

---

## Subagent overall assessment

The codebase is well-structured and mostly faithful to the paper. The core training loop, baseline implementations, and metrics are correct. However, there are three material issues: (1) a unit mismatch on the "%+Ret" metric (percentages in output vs. fractions in paper), (2) silent exception swallowing in the evaluation pipeline that hides training/data failures, and (3) a noisy end-of-episode flush for A2C.

**[MAIN]: Agree.** This subagent caught the two highest-impact issues in its scope (B2, B3 in the overall review).

---

## Findings

### metrics.py:53 — `%+Ret` unit mismatch

> Subagent: `%+Ret` returned as percentage (0–100) but paper's Exhibit 2 shows fractions (0–1). When CSV is formatted with `float_format="%.4f"`, a value like "52.34" is printed alongside metrics like "0.0234" (Sharpe), creating ambiguity about units.
>
> Severity: spec-deviation (breaks Exhibit 2 reproduction).

**[MAIN]: Agree. Promoted to B3 in the overall review.** Trivial fix. I'd classify this as a bug (units), not just a spec deviation, since it silently mismatches the paper's reported format.

---

### evaluate.py:181–183, 212–214 — Silent exception swallowing

> Subagent: Baseline and RL agent training failures are caught and silently stored as empty Series. No stack trace; only short error message printed. Empty Series propagate downstream, producing NaN or zero metrics without warning.
>
> Severity: bug.

**[MAIN]: Strongly agree. Promoted to B2 in the overall review.** Either remove the try/except entirely or collect failures into a list and print a summary at the end. For a research experiment the default should be to surface problems loudly.

---

### trainer.py:118–121 — A2C end-of-episode flush with `batch_size=1`

> Subagent: Temporarily sets `batch_size=1` to train remaining steps. Allows a single-step batch to update the network after normal batch training, producing high-variance, non-representative loss spikes.
>
> Severity: nit.

**[MAIN]: Agree.** Cleaner alternatives: (a) drop the leftover buffer entirely at episode end; (b) pad with no-op transitions; (c) make the flush optional via a config flag. (a) is simplest and I'd recommend it.

---

### baselines.py:54 — Redundant clip on MACD signal

> Subagent: `np.clip(macd_signal, -1.0, 1.0)` is unnecessary. The φ transformation already bounds values to ~[-0.5, 0.5], so clipping is defensive but redundant.
>
> Severity: nit.

**[MAIN]: Agree.** Harmless; delete if cleaning up.

---

### baselines.py:79 — Silent truncation

> Subagent: `n = min(len(positions), len(prices)-1)` silently truncates excess positions. If a baseline strategy generates more positions than available prices, the function returns fewer rewards than positions without warning.
>
> Severity: nit.

**[MAIN]: Agree.** In practice `positions` always comes from `_get_symbol_split(...)` which shares the same filter used for prices, so they're the same length. The guard is defensive. Consider an `assert` instead of silent truncation to catch real bugs.

---

### trainer.py:134–141 — `_compute_sharpe` on daily rewards

> Subagent: `mean(r) / std(r) * √252`. This is the Sharpe of the *daily reward* time series (already vol-scaled trade returns), which differs from Sharpe of e.g. daily portfolio returns. Matches metrics.py convention.
>
> Severity: nit.

**[MAIN]: Agree, and worth noting.** This is only used for early-stopping decisions during training (val Sharpe). Since the reward already includes vol scaling and costs, this is effectively a risk-adjusted return of the strategy — reasonable for model selection. Consistent with `metrics.py` computations used for the final report.

---

### evaluate.py:156 — Hardcoded data-fetch date range

> Subagent: Uses hardcoded end date "2025-12-31", matching TEST_END in config. No risk of look-ahead bias, but inflexible.
>
> Severity: nit.

**[MAIN]: Agree.** Refactor to `min(TRAIN_START, ...), max(..., TEST_END)` from config. Low priority.

---

### evaluate.py:248 — Portfolio aggregation

> Subagent: Uses outer join + `.mean(axis=1).dropna()`. Correct for staggered symbol start dates.
>
> Severity: nit.

**[MAIN]: Agree.** Verified: `pd.DataFrame.mean(axis=1)` is NaN-safe by default (`skipna=True`), so the mean is taken over whichever symbols are live on each date. `.dropna()` drops only all-NaN rows. Correct.

---

### UNSURE — `trainer.py:107–108` — PG last-reward inclusion

> Subagent: Loop exits when `done=True`, but reward is appended **before** checking `done`. So terminal reward IS included in `self.agent.rewards` before `train_episode()` is called. Correct for REINFORCE.

**[MAIN]: Agree.** Verified by tracing: `env.step` returns `done=True` with the reward from the terminal transition (line 175 in env.py), the trainer's `store` call writes that reward, then the loop exits and `train_episode` runs on the full trajectory including terminal. Correct.

---

### UNSURE — `evaluate.py:_portfolio_vol_scale`

> Subagent: Expanding window vol targeting applied after portfolio aggregation. First 60 days unscaled (insufficient history after shift). Thereafter, realized vol from days 0–t scaled to apply to day t+1. This matches the paper's "portfolio-level volatility targeting" but the paper doesn't detail the exact implementation.

**[MAIN]: Agree that it's defensible, but two things to flag that the subagent didn't:**
1. **Expanding window vs rolling.** Paper says "portfolio-level volatility targeting" without specifying. Expanding window gives increasingly stable scale as more data accumulates; a rolling window (e.g. 60-day) would adapt more to regime shifts. Both are defensible.
2. **Scale = 1 for first 60 days is a soft cliff.** On day 60 the scale jumps from 1.0 to whatever the computed scale is. Makes cumulative-return plots slightly discontinuous around day 60. Cosmetic.

No fix needed; just be aware of the convention.

---

## Findings not covered by the subagent

### metrics.py — MDD via cumulative SUM, not cumulative PRODUCT

The subagent's checklist noted this in passing but didn't annotate. For small daily returns (few % per day), cumsum ≈ cumprod, so MDD is a reasonable approximation. But under large single-day moves (e.g. vol-scaled with ann_vol_t at the 1e-8 floor — see env.py:143), cumsum and cumprod diverge. Worth a comment clarifying the approximation.

### evaluate.py:140–147 — UNIVERSE override creates "other" asset class

When the user passes `--symbols FOO` with `FOO` not in `UNIVERSE`, the code appends `{"symbol": "FOO", "asset_class": "other"}`. An "other" group silently appears in the portfolio aggregation. Validate or warn.

### evaluate.py:_plot_cumulative (lines 303–333) — Missing axis labels/sharex

Stacked subplots per group. `axes[-1].set_xlabel("Trading Day")` on the bottom only. The x-axis is integer trading days, not dates, so all subplots are comparable, but dating the x-axis (from the portfolio return index) would be more useful for interpretation. Minor.

### trainer.py:46–68 — Training prints every epoch

For 200 epochs × ~45 symbols × 3 agents = ~27,000 lines. Consider `print` only when val Sharpe improves, or gate on `eval_every`. Nit.

### baselines.py — Baselines don't use the env

The three baseline position functions (`long_only`, `sign_r`, `macd_signal`) return position arrays; `compute_baseline_rewards` applies the env's reward formula by hand. This duplicates the reward logic in `env.step`. If `env.step` ever changes, `baselines.compute_baseline_rewards` must track it. Consider a helper (`apply_reward(positions, prices, ewm_vol, ...)`) that both call. Low priority.

---

## Summary severity distribution after annotations

- Bugs: 2 (%+Ret units — B3; silent exception swallowing — B2).
- Spec deviations: 0 (the noisy A2C batch_size=1 flush is really a nit).
- Nits: 7 (trainer A2C flush; MACD clip; baselines truncate; hardcoded dates; logging volume; MDD cumsum vs cumprod; reward-logic duplication).
