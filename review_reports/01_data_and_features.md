# Subagent 1 — Data & Features Review

**Scope.** `config.py`, `data.py`, `features.py`
**Prompt.** `../review_prompts/1_data_and_features.md`

> Annotations by the main reviewer are prefixed with **[MAIN]:** and appear inline under each finding. Items without an annotation are accepted as-is.

---

## Subagent overall assessment

The implementation closely follows the paper's feature definitions with strong adherence to formulas and window specifications. The code is well-structured and most features appear correct. However, there are three notable issues: (1) a critical off-by-one error in the `window_ready` calculation that gates features prematurely, (2) an EWM adjustment parameter mismatch that could subtly affect volatility estimates, and (3) unused intermediate return calculations that should be cleaned up or their purpose clarified.

**[MAIN]:** Partially disagree. Issue (1) as stated is wrong — the warmup chain for MACD is parallel, not sequential. See annotations below.

---

## Findings

### features.py:96 — `window_ready` off-by-one error

> Subagent: Line 96 computes `min_needed = max(252, max_ema_span + MACD_PRICE_STD_WINDOW - 1 + MACD_NORMALIZATION_WINDOW - 1)` but the `-1` terms are incorrect. The MACD formula requires `max_ema_span` (for EMA warm-up) + `MACD_PRICE_STD_WINDOW - 1` (63-day rolling std window, needs 63 bars) + `MACD_NORMALIZATION_WINDOW - 1` (252-day q-std, needs 252 bars), but consolidating: you need the EMA to be ready (max_ema_span bars), *then* MACD_PRICE_STD_WINDOW bars after that for the first price std, *then* MACD_NORMALIZATION_WINDOW bars for the q-std. The formula should be `max_ema_span + MACD_PRICE_STD_WINDOW + MACD_NORMALIZATION_WINDOW - 2`, not `-1` on both terms. This causes the flag to be set one row too early.
>
> Severity: bug (gates MACD features ~1 bar before they're valid).

**[MAIN]: Disagree — the subagent's warmup reasoning is wrong.** EMA and the 63-day rolling price std operate on the *same* price series in parallel, not sequentially. They both start producing valid output after ~96 rows (max span, since 96 > 63). Then the 252-day std of `q` needs 252 valid `q` observations, starting once `q` becomes valid. Correct first-valid index is 95 + 251 = 346 (0-indexed). The code sets `window_ready=True` at row 347, i.e. one row more conservative than strictly needed — harmless. Not a bug. Both the subagent's body ("bar off too early") and UNSURE section ("one short") got the direction confused.

**Actual status:** slightly conservative (~1 row), negligible. Not worth fixing unless you care about the very first usable day.

---

### features.py:58 — EWM volatility `adjust=True` default

> Subagent: Line 58 uses `.ewm(span=60, min_periods=60).std()`, which defaults to `adjust=True`. The paper specifies a 60-day EWM volatility but does not explicitly state whether to use recursive (adjust=False) or recursive with lookback adjustments. The default `adjust=True` computes the recursion with an implicit pre-window, which differs from a pure 60-day recursive formula. For consistency with standard implementations and given the 60-day span context, `adjust=False` may be more appropriate.
>
> Severity: spec-deviation.

**[MAIN]: Agree.** Worth switching to `adjust=False` if you care about matching the pure recursive formula the paper implies. Subtle effect in practice. Low priority.

---

### features.py:69–70 — Intermediate return columns stored but not used

> Subagent: Lines 69-70 compute `ret_{h}` for each horizon (21, 42, 63, 252 days) and store it, but only `ret_{h}_vol` (normalized by volatility) is included in `FEATURE_COLS` (lines 21-32). The raw returns are dead code or stored for external use (e.g., reward computation).
>
> Severity: nit (no functional impact, clarification needed).

**[MAIN]: Partially correct.** `ret_252` is consumed by `baselines.sign_r` (line 48), so not dead. The other horizons (`ret_21`, `ret_42`, `ret_63`) are unused. Either delete them or add a comment noting the `ret_252`/baseline dependency.

---

### features.py:48 — RSI scaling equivalence claim

> Subagent: Line 48 comment states `rs / (1 + rs)` is "equivalent to RSI/100". This is mathematically correct: RSI = 100 * rs / (1 + rs), so dividing by 100 yields `rs / (1 + rs)`. The implementation is correct.
>
> Severity: none (verified).

**[MAIN]: Agree.** Correct.

---

### features.py:76–79 — MACD formula matches paper

> Subagent: Code computes `q = (ema_s - ema_l) / price_std` and `macd = q / q_std`, which matches paper Eq. 3 exactly. Rolling windows (63-day price std, 252-day q-std) and EWMA usage (not SMA) align with paper. The `_phi()` function (line 35-37) correctly implements φ(x) = x·exp(−x²/4)/0.89 and is applied to each MACD before averaging (line 87).
>
> Severity: none (correct).

**[MAIN]: Agree.** Verified.

---

### features.py:64 — `norm_close` formula

> Subagent: Line 64 computes `ret_252 / ann_vol`, where `ret_252 = close / close.shift(252) - 1` is the 252-day return and `ann_vol` is the annualized EWM volatility (daily std × √252). The paper says normalize by σ_t·√252. This matches.
>
> Severity: none (correct).

**[MAIN]: Agree.**

---

### features.py:67–70 — Vol-adjusted returns horizons

> Subagent: Code uses 21, 42, 63, 252 days, matching the paper (1, 2, 3, 12 months on business days). Formula `cum_ret / ann_vol` is correct.
>
> Severity: none.

**[MAIN]: Agree.**

---

### data.py:47–50, 67–70 — yfinance setup and fallback logic

> Subagent: The code calls `yf.download()` with `auto_adjust=False` (preserves both `close` and `adj_close`), normalizes column names, and falls back to `close` if `adj_close` is missing (line 70). The tz-localize on line 72 correctly strips timezone info (`dt.tz_localize(None)`). Multi-symbol indexing via `raw[sym]` assumes either a single-symbol result or a MultiIndex; the code handles both (line 55). No cache key collision risk: hash includes sorted symbols + start + end.
>
> Severity: none.

**[MAIN]: Agree.** One minor follow-on — the cache doesn't version the schema. If `fetch_bars` output format ever changes, stale parquet files won't be invalidated. Consider bumping the key format. Not urgent.

---

### config.py:20–25 — Train/val/test splits

> Subagent: Paper uses 2011–2019 for futures (8 years); repo uses 2005–2019 (15 years) with a train/val/test split at 2016/2019. This is a known deviation (repo uses ETFs, not futures) acknowledged elsewhere.
>
> Severity: none.

**[MAIN]: Agree** — acknowledged in `TODO.md`. Worth noting that `TEST_END="2025-12-31"` is in the future relative to the paper's test window — fine for ETF replay, just be aware test data includes post-COVID and post-2022 regimes the paper didn't touch.

---

### config.py:6–12 — Feature windows match paper

> Subagent: `VOLATILITY_SPAN=60`, `RSI_WINDOW=30`, `MACD_WINDOWS=((8,24),(16,48),(32,96))`, `MACD_PRICE_STD_WINDOW=63`, `MACD_NORMALIZATION_WINDOW=252`, `RETURN_HORIZONS=(21,42,63,252)`. All align.
>
> Severity: none.

**[MAIN]: Agree.**

---

### features.py:94–99 — UNSURE block

> Subagent: A 96-bar EMA requires 96 bars; then you need 63 bars for the price std (so 96 + 63 = 159 bars total); then you need 252 bars of the q series (so 159 + 252 = 411 bars). The formula yields 410, one short.
>
> Severity: bug.

**[MAIN]: Disagree (same issue as main body).** The "96 + 63 = 159" chain assumes you need 63 more bars *after* EMA is ready to compute the price std, but the price std is computed on *prices*, not on EMA output. Both the EMA and the price std use raw prices in parallel. Correct first-valid row is 346; code uses 347 (1 row conservative). See the main body annotation.

---

## Main reviewer's additional items (not in subagent report)

- **`config.py:28–78`** — Universe hardcoded as a list of dicts. If you want per-asset-class model training later (as TODO.md proposes), the per-class grouping would be easier with a small helper like `by_class()`.
- **`data.py:43–50`** — `threads=True` on `yf.download` combined with a hash-based cache: fine, but if yfinance returns partial data intermittently, the cache locks in the partial result. Consider invalidating on row-count sanity check.

---

## Summary severity distribution after annotations

- Bugs: 0 (subagent claimed 1; disagreed).
- Spec deviations: 1 (EWM `adjust` default).
- Nits: 2 (unused `ret_{h}` raw columns; minor cache versioning).
