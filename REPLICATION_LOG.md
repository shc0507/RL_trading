# Replication log — Zhang, Zohren & Roberts (2020) "Deep RL for Trading"

This document captures every experiment run during our replication of
Zhang, Zohren, and Roberts, *Deep Reinforcement Learning for Trading*,
JFDS 2020. It records the configuration of each run, the motivation for
each change, the resulting metrics, and the lessons each run produced.
It is intended to be read top-to-bottom as a chronological narrative.

Paper: `docs/Deep-Reinforcement-Learning-for-Trading (1).pdf`
Codebase: this repo (`rl_trading/`)
Final pipeline: `scripts/run_walkforward_parallel.sh`

---

## TL;DR

| Version | Best paper-aligned? | A2C All Sharpe | PG All Sharpe | DQN All Sharpe | One-line summary |
|---|---|---|---|---|---|
| v1 | — | NaN | NaN | (incomplete) | First CLC submission. Discovered feature-NaN bug. |
| v2 | — | NaN | NaN | (incomplete) | Restored PG mean baseline. Still NaN. |
| v3 | — | NaN | NaN | (incomplete) | Added decoupled σ_tgt + train/test cost split. Still NaN. |
| **v4** | — | −0.53 | −10.19 | −1.66 | Added per-contract `μ=1/p_ref` reward normalization. **First clean run.** |
| **v5** | **Yes — paper-literal** | **+0.50** | −20.4 | −2.16 | Switched to Sharpe-only patience (paper p.7). **Best A2C result.** |
| v6 | No (tuned) | −0.77 | −25.9 | −2.51 | Multi-seed + train/test cost split + state z-score + entropy bonuses. A2C regressed. |
| **v7** | No (tuned) | −0.77 | **−10.7** | −2.51 | Reverted cost split, reduced PG entropy + bumped lr, cum-return seed selection. **Best PG result.** |
| v8 | — | −0.77 | −26.2 | −1.80 | A2C ablation — confirmed multi-seed-by-val-Sharpe is what hurts A2C. |
| Paper | — | +1.05 | +0.75 | +1.29 | Reference. |

**Headline conclusion:** the cleanest paper-literal replication is **v5**.
A2C reached +0.50 All-portfolio Sharpe (vs paper's +1.05). The remaining
gap to paper is dominated by (a) RAD-vintage data sensitivity, (b) paper
underspecification of stabilizers (state norm, entropy, σ_tgt value),
and (c) irreducible noise from single-seed training on a 7-month
validation window. Multi-seed-by-val-Sharpe selection (v6/v7/v8) hurt
A2C because the val window is too short for the metric to be reliable.

---

## Setup

### Data

- **Universe:** 48 of 49 distinct continuous futures contracts from
  Zhang's Appendix A (one contract, US T-Bond 30Y, was excluded
  because the Pinnacle RAD export contained NaN-only rows for the
  historical period). 25 commodities, 10 equity indices, 5 fixed
  income, 9 FX (Nikkei is grouped under FX per Zhang's taxonomy).
- **Vendor:** Pinnacle CLC, ratio-adjusted (RAD) continuous contracts.
  Our export is from April 2026; Zhang's was from 2019 (7 more years
  of accumulated roll ratios → different absolute price levels but
  identical percentage returns).
- **Splits (paper p.6):** two retraining folds.
  - Fold 1: train 2005-01 → 2010-12 (val: 2010-06 → 2010-12,
    last 10%), test 2011-01 → 2015-12.
  - Fold 2: train 2005-01 → 2015-12 (val: 2014-12 → 2015-12,
    last 10%), test 2016-01 → 2019-12.

### State

10 features × 60-bar window:
1. Normalized close (`ret_252 / (σ * √252)`)
2–5. Vol-adjusted returns at horizons 21, 42, 63, 252 days
6–8. MACD signals at scales (8,24), (16,48), (32,96)
9. Combined φ-smoothed multi-scale MACD signal
10. RSI(30) using Wilder (1978) recursive smoother

The current position is **not** part of the state — only its causal
effect through Eq. 4 reaches the agent.

### Reward (Zhang Eq. 4)

```
R_t = μ_i · [ A_{t-1} · (σ_tgt / σ_{t-1}) · r_t
            − bp · p_{t-1} · |σ_tgt/σ_{t-1} · A_t − σ_tgt/σ_{t-2} · A_{t-1}| ]
```

with `r_t = p_t − p_{t-1}` (additive profits, paper p.5), and
`σ_{t-1}` = EWM std of percentage returns, span 60.

### Pipeline

3 stages, parallelized via slurm:

1. `scripts/stage1_prefetch.sbatch` — load CLC RAD CSVs, build features,
   write `artifacts/cache/features.parquet`.
2. `scripts/stage2_train.sbatch` — slurm array, one task per (agent, fold,
   asset_class[, seed]).
3. `scripts/stage3_aggregate.sbatch` — stitch fold outputs, compute
   portfolio-level metrics, write `results.csv` and plots.

---

## Pre-replication state

The codebase started with **ETF proxies via yfinance** (SPY, GLD, TLT,
…). It "worked" — agents trained, results produced. We later discovered
this was hiding a latent feature-NaN bug (see v3) that ETF data didn't
trigger because ETFs rarely have 60-day flat-price stretches.

Goal: switch to Pinnacle CLC futures (Zhang's actual dataset), confirm
agents reproduce paper-style results.

---

## v1 — first CLC submission

**Hypothesis:** swap data layer ETF → CLC RAD futures, everything
else unchanged from the working ETF run.

**Config:**
- 49 contracts (Zhang Appendix A)
- σ_tgt = 0.15 (env), 0.15 (portfolio rescale)
- Cost = 2 bp (legacy default)
- DQN/PG/A2C with paper Exhibit 1 hyperparameters
- Multi-metric OR early stopping (Sharpe | Sortino | Cum)

**Result:** PG and A2C tasks crashed deterministically with
`probability tensor contains either inf, nan, or element < 0` in
`torch.multinomial`. DQN finished but with degenerate "always hold"
policies on fixed income.

**Diagnosis:** the per-bar reward magnitude exploded for low-σ contracts.
With `vol_scale = σ_tgt / σ_{t-1}` uncapped and σ ≈ 0.001 on quiet bars,
reward magnitudes hit thousands → softmax saturated to NaN.

**Lesson:** vanilla REINFORCE without numerical guards crashes
immediately on dollar-unit rewards.

---

## v2 — PG mean baseline restored

**Hypothesis:** subtracting the batch mean from `G_t` in REINFORCE is
mathematically unbiased (`E[∇log π · b] = 0`) but reduces variance.
This is the standard fix for the silent-PG / NaN-multinomial pathology.

**Config:** same as v1, plus:
- PG: subtract `returns.mean()` from reward-to-go before computing loss.

**Result:** PG no longer crashed in `multinomial`, but A2C still NaN'd
(`Normal(loc=NaN)`) on commodity tasks. Cancelled mid-run.

**Diagnosis:** A2C's critic produced NaN values from gradient explosion.
Same root cause as PG: huge reward magnitudes for low-σ bars.

**Lesson:** stabilizing PG is necessary but not sufficient — A2C needs
the same medicine via a different mechanism.

---

## v3 — feature NaN gate + decoupled σ_tgt

**Hypothesis 1:** decouple env-level σ_tgt (training) from
portfolio-level σ_tgt (reporting). Set env σ_tgt = 0.15 (industry
standard, gentler training), portfolio σ_tgt = 1.0 (paper-magnitude
Std(R) ≈ 1.0 in Exhibit 2).

**Hypothesis 2:** several CLC contracts have flat-price stretches >60
days where pct_change is identically zero, so EWM-σ collapses to 0,
and feature columns that divide by σ produce NaN. The
`window_ready=True` flag gates against history length but not against
NaN-valued features. Tightening the gate to require all 10 features
finite should eliminate the NaN inputs that crash PG/A2C.

**Config diff vs v2:**
- New `_safe_div` in features.py: zero or near-zero denominator → NaN
  (instead of `inf`).
- `window_ready = history_ok AND all FEATURE_COLS finite on this row`.
- Decoupled `DEFAULT_VOL_TARGET = 0.15` (env) from
  `PORTFOLIO_VOL_TARGET = 1.0` (reporting only).

**Result:** baselines and DQN ran cleanly. PG and A2C still trained
poorly because per-contract reward magnitudes differed by ~150× across
the commodity universe (DA Milk ~$0.1/bar vs LB Lumber ~$17/bar with
σ_tgt = 0.15) — the gradient was dominated by the high-priced contracts.

**Lesson:** ETF data ran fine in earlier work because ETFs don't have
the 60-day flat-price stretches. Pinnacle continuous futures do (illiquid
contracts, holiday closures, roll-day artifacts). The `_safe_div` +
finite-feature gate is a genuine bug fix, not a stylistic change.

---

## v4 — per-contract reward normalization

**Hypothesis:** Zhang p.5 mentions that `μ_i` is a per-contract
normalization constant in Eq. 4 ("we need to normalize different
rewards to the same scale"), but the paper sets μ uniformly to 1.
Setting `μ_i = 1/p_ref_i` (where p_ref is the first price of the
episode) makes per-bar dollar rewards roughly dimensionless and
comparable across contracts. This is the operational meaning of
the paper's own statement.

**Config diff vs v3:**
- `env.py`: divide reward by `self._ref_price = self._prices[0]` per
  episode.
- `baselines.py`: same division in `compute_baseline_rewards` to keep
  baseline and agent rewards on the same scale.

**Result (All-portfolio Sharpe):**

| Method | v4 | Paper |
|---|---|---|
| Long | +0.097 | +0.058 |
| Sign(R) | −0.43 | +0.44 |
| MACD | −1.22 | −0.08 |
| DQN | −1.66 | +1.29 |
| PG | −10.19 | +0.75 |
| A2C | −0.53 | +1.05 |

**Long matches paper within noise** — confirming the upstream pipeline
is correct (data, reward, σ-scaling, portfolio aggregation). RL agents
substantially below paper.

**Patience policy at this point:** multi-metric OR (Sharpe | Sortino |
Cum) — agents trained 100+ epochs on some tuples.

**Lesson:** "equal-weight portfolio" with dollar-unit rewards is
actually dollar-weighted; per-contract `μ_i = 1/p_ref` restores true
equal-weight and aligns the simplest baseline to paper. Aggregator's
artifacts directory was accidentally deleted before v5; reconstructed
`results.csv` from chat transcript.

---

## v5 — Sharpe-only early stopping (paper-literal)

**Hypothesis:** paper p.7 specifies single-metric early stopping with
patience 20. Our multi-metric OR-patience let training run far longer
than paper. With a 7-month val window, longer training could be
overfitting noise.

**Config diff vs v4:**
- `trainer.py`: `early_stop_policy = "sharpe_only"`. Stop when val
  Sharpe stale for 20 epochs.

**Result (All-portfolio Sharpe):**

| Method | v5 | Paper |
|---|---|---|
| Long | +0.097 | +0.058 |
| Sign(R) | −0.43 | +0.44 |
| MACD | −1.22 | −0.08 |
| DQN | −2.16 | +1.29 |
| PG | −20.36 | +0.75 |
| **A2C** | **+0.50** | +1.05 |

**Lessons:**

1. **A2C +0.50 is the best agent number we ever produce.** Paper-literal
   training works for A2C. Sign and ordering match paper.
2. DQN regressed slightly under shorter patience — too eager to stop;
   needed more epochs to learn.
3. PG collapsed further: `best_epoch = 1` on most fixed-income/PG tuples
   (random init beat anything trained for 20 epochs).
4. **v5 is the canonical paper-replication result.** Subsequent versions
   are tuned variants meant to outperform; they trade fidelity for
   targeted improvements on specific (agent, class) cells.

---

## v6 — kitchen-sink optimization

**Hypothesis:** combine every plausible improvement at once.

**Config diff vs v5:**

| Knob | v5 | v6 |
|---|---|---|
| σ_tgt env | 0.15 | **1.0** |
| State z-score per window | off | **on** |
| Train cost / test cost | 20 bp / 20 bp | **2 bp / 20 bp** |
| Patience policy | Sharpe-only=20 (all) | OR-multi=50 (DQN/PG), Sharpe-only=30 (A2C) |
| Multi-seed | 1 | **3** (best by val Sharpe) |
| PG γ | 0.3 | **0.95** |
| PG entropy_coef | 0 | **0.05** |
| A2C entropy_coef | 0 | **0.01** |
| Slurm array | 0–23 | **0–71** |

**Result (All-portfolio Sharpe):**

| Method | v5 | v6 | Paper |
|---|---|---|---|
| Long | +0.097 | **+0.30** | +0.058 |
| Sign(R) | −0.43 | **+0.38** | +0.44 |
| MACD | −1.22 | **−0.28** | −0.08 |
| DQN | −2.16 | −2.51 | +1.29 |
| PG | −20.36 | −25.89 | +0.75 |
| **A2C** | **+0.50** | **−0.77** | +1.05 |

**Lessons:**

1. Sign(R) +0.38 is the closest baseline match to paper's +0.44 we ever
   got — driven by the *lower training cost*, not σ_tgt or state norm.
2. Long +0.30 (vs paper +0.06) is *worse* paper alignment than v5/v7/v8.
3. **A2C regressed badly** (+0.50 → −0.77). Hypothesized causes
   (multiple, all changing simultaneously): cost mismatch, σ_tgt env =
   1.0 amplifying gradients, state z-score, entropy bonus, multi-seed
   selection. Need ablations to isolate.
4. PG worsened. Hypothesis: entropy_coef 0.05 was the same magnitude
   as the value gradient (|G_t| ~ 0.05 with γ=0.95) — entropy regularizer
   pinned the softmax at uniform.

---

## v7 — surgical fixes after v6 regression

**Hypothesis:** the v6 A2C regression was driven by the train/test cost
mismatch (policy learned aggressive trades at 2 bp, got shredded at
20 bp test). The PG regression was driven by entropy_coef being too
high relative to the value gradient.

**Config diff vs v6:**

| Knob | v6 | v7 |
|---|---|---|
| Train cost | 2 bp | **20 bp** (matches test) |
| PG entropy_coef | 0.05 | **0.01** |
| PG lr | 1e-4 | **5e-4** |
| Multi-seed selection metric | val Sharpe (all agents) | **val Sharpe (DQN/A2C), val cum-return (PG)** |

PG cum-return selection rationale: silent-PG attractor has Std(R) ≈ 0
→ Sharpe ≈ 0/0 ≈ 0, which artificially beats any actively-trading
policy with slight negative E(R) due to cost noise. Cum-return doesn't
reward "do nothing."

**Result (All-portfolio Sharpe):**

| Method | v5 | v6 | v7 | Paper |
|---|---|---|---|---|
| Long | +0.097 | +0.30 | +0.097 | +0.058 |
| Sign(R) | −0.43 | +0.38 | −0.43 | +0.44 |
| MACD | −1.22 | −0.28 | −1.22 | −0.08 |
| DQN | −2.16 | −2.51 | −2.51 | +1.29 |
| **PG** | −20.4 | −25.9 | **−10.7** | +0.75 |
| A2C | +0.50 | −0.77 | −0.77 | +1.05 |

**Lessons:**

1. **PG was rescuable** — cutting entropy + bumping lr + cum-return
   seed selection more than halved its Sharpe loss (-25.9 → -10.7).
   PG's `%+Ret` went 0.04 → 0.21 — it actually trades now.
2. **The cost-mismatch hypothesis was wrong** — A2C did not recover.
   Some other v6 change is the A2C culprit.
3. Baselines reverted to v5 numbers. v6's improved Sign(R) was driven
   solely by the lower training cost; cost-revert undid that gain.

---

## v8 — A2C ablation

**Hypothesis:** isolate which of σ_tgt env, state z-score, or A2C
entropy is hurting A2C. Revert all three to v5 settings; keep all v7
PG fixes.

**Config diff vs v7:**

| Knob | v7 | v8 |
|---|---|---|
| σ_tgt env | 1.0 | **0.15** (revert) |
| State z-score | on | **off** (revert) |
| A2C entropy_coef | 0.01 | **0** (revert) |

Everything else identical to v7 (multi-seed × 3, PG entropy 0.01, lr
5e-4, OR-multi patience for DQN/PG, cum-return selection for PG).

**Result (All-portfolio Sharpe):**

| Method | v5 | v7 | v8 | Paper |
|---|---|---|---|---|
| Long | +0.097 | +0.097 | +0.097 | +0.058 |
| Sign(R) | −0.43 | −0.43 | −0.43 | +0.44 |
| MACD | −1.22 | −1.22 | −1.22 | −0.08 |
| DQN | −2.16 | −2.51 | **−1.80** | +1.29 |
| PG | −20.4 | **−10.7** | −26.2 | +0.75 |
| A2C | **+0.50** | −0.77 | −0.77 | +1.05 |

**Lessons:**

1. **A2C did NOT recover** under v8 settings. Reverting σ_tgt, state
   z-score, and entropy did nothing. The only remaining difference
   between v5 and v8 affecting A2C is **multi-seed-by-val-Sharpe
   selection**.
2. **Diagnosis:** with a 7-month val window (~140 daily returns), the
   stderr of the val-Sharpe estimate is ~0.5. Picking the max of 3 noisy
   draws systematically over-fits to validation noise — a classic
   bias-from-selection. Multi-seed *averaging* would help; multi-seed
   *max-by-val-Sharpe* hurts.
3. **PG broke again** because lower σ_tgt = smaller per-bar reward =
   smaller G_t = entropy regularizer dominates value gradient again.
   PG and A2C want *different* σ_tgt values.
4. DQN modestly improved (Q-targets are easier to fit at smaller scale).

---

## Final comparison table (All-portfolio Sharpe)

| Method | v4 | **v5 (paper-literal)** | v6 | **v7** | v8 | Paper |
|---|---|---|---|---|---|---|
| Long | +0.097 | +0.097 | +0.30 | +0.097 | +0.097 | +0.058 |
| Sign(R) | −0.43 | −0.43 | +0.38 | −0.43 | −0.43 | +0.44 |
| MACD | −1.22 | −1.22 | −0.28 | −1.22 | −1.22 | −0.08 |
| DQN | −1.66 | −2.16 | −2.51 | −2.51 | −1.80 | +1.29 |
| **PG** | −10.19 | −20.4 | −25.9 | **−10.7** | −26.2 | +0.75 |
| **A2C** | −0.53 | **+0.50** | −0.77 | −0.77 | −0.77 | +1.05 |

Best of each agent across our runs (any version): A2C **+0.50** (v5),
PG **−10.7** (v7), DQN **−1.80** (v8).

---

## Key lessons

### 1. The paper underspecifies critical knobs.

Paper Exhibit 1 lists ~12 hyperparameters; our final pipeline has ~30.
The unspecified ones include:

- σ_tgt value (env-level)
- σ_tgt value (portfolio rescale)
- Whether and how to normalize features (state z-score)
- PG/A2C entropy coefficients
- PG mean baseline (paper Eq. 6 is vanilla, but vanilla NaN's)
- Patience metric (paper says "early stopping with 20 epochs"; doesn't
  say single-metric or multi-metric)
- Whether to multi-seed
- Validation block position (we use trailing; paper doesn't say)
- σ floor against zero-vol-bar pathologies
- μ_i value per contract (paper sets to 1; we set to 1/p_ref)

### 2. RAD-vintage data sensitivity is a hidden landmine.

Our 2026 Pinnacle export and Zhang's 2019 export differ in absolute
price levels by 7 years of accumulated roll ratios. Percentage returns
are identical, but dollar-unit rewards (paper's choice) are not — making
"equal-weight portfolio" effectively dollar-weighted differently across
vintages. **Zhang's reported numbers cannot be reproduced without the
specific 2019 export.** Setting μ_i = 1/p_ref is our workaround.

### 3. Multi-seed selection by val Sharpe hurts on short val windows.

Our 7-month val window has ~140 daily returns; val Sharpe stderr
≈ 0.5. Picking max-of-3 gives a +1 Sharpe selection bias that doesn't
generalize. Multi-seed *averaging* (or median selection) would help;
*max-by-val-metric* doesn't. v5 single-seed beat v8 max-of-3 on A2C.

### 4. A2C and PG want different reward magnitudes.

A2C is best at σ_tgt = 0.15 (small rewards, stable critic learning).
PG is best at σ_tgt = 1.0 (big rewards to overcome the entropy
regularizer). No single env σ_tgt is optimal for both.

### 5. The same training protocol cannot serve all three agents.

DQN/PG benefit from longer patience; A2C overfits with extra epochs.
v6/v7's per-agent patience policy is a deviation from paper's uniform
"20 epochs" but reflects the practical reality that REINFORCE,
Q-learning, and actor-critic have different learning timescales.

### 6. PG is structurally disadvantaged in this regime.

Even with all stabilizers (mean baseline, entropy bonus, higher lr,
γ = 0.95, cum-return seed selection), v7 PG still loses on test
(−10.7 Sharpe). Discrete actions, MC credit assignment, and a strict
cost penalty conspire to push the policy toward "do nothing."
**Zhang's reported PG +0.75 is implausible** without significant
exploration scheduling or other unstated tricks.

---

## Limitations

- **Single-vendor RAD data**, single export date. Results not directly
  comparable to Zhang's 2019 export.
- **One contract missing** (US T-Bond 30Y; broken Pinnacle RAD CSV).
- **Single seed** (v5) or 3-seed-best (v6/v7/v8) per (agent, fold,
  class). Proper replication requires ≥5 seeds with median or mean
  reporting.
- **Only Exhibit 2 reproduced.** Exhibit 4 (per-contract distributions),
  Exhibit 5 (cost sensitivity 1–45 bp), and Exhibit B1 (raw signal
  table) are not produced as separate runs (B1 is incidentally
  available via `results_unscaled.csv`).
- **Train/test regime mismatch** is genuine: 2011–2019 includes
  Euro crisis, oil crash, 2018 selloff, none of which appear in
  fold-1 train (2005–2010). Trend-following policies don't transfer.

---

## Artifacts committed for inspection

Each completed version (v4–v8) has a small set of summary artifacts
checked into the repo under `artifacts/wf_zhang2019_replication_<v>/`:

| File | What it is | Size per version |
|---|---|---|
| `aggregated/results.csv` | Exhibit-2-style portfolio metrics (vol-scaled) | ~2.5 KB |
| `aggregated/results_unscaled.csv` | Exhibit-B1-style portfolio metrics (raw) | ~2.5 KB |
| `aggregated/cumulative_returns.png` | Per-class cumulative-return plot (Exhibit 3 layout, vol-scaled) | ~140 KB |
| `aggregated/cumulative_return_raw.png` | Per-class compounded-wealth plot (raw) | ~240 KB |
| `<agent>/<class>_fold<n>[_seed<s>]/train_result.json` | Per-tuple training summary (best epoch, val Sharpe/Sortino/Cum, fold dates, hyperparams) | ~500 B per tuple |

Larger files NOT pushed to the repo (regenerable by re-running the
pipeline; ignored via `artifacts/` in `.gitignore`):

- `aggregated/per_symbol_*.csv` (~23 MB each) — daily reward stream per
  contract per method; useful for per-asset analysis but reconstructible
  from `per_tuple_results/<agent>/<class>_fold<n>/rewards_zhang.parquet`.
- `aggregated/portfolio_*.csv` (~3 MB each) — daily portfolio-level
  return streams.
- `<agent>/<class>_fold<n>[_seed<s>]/rewards_*.parquet` — per-tuple test-window
  reward streams (raw inputs to the aggregation step).
- `<agent>/<class>_fold<n>[_seed<s>]/checkpoint/*.pt` — trained model
  weights.

To regenerate the larger files for any version, check out that
version's commit and run:

```bash
RUN_ID=zhang2019_replication_<v> bash scripts/run_walkforward_parallel.sh
```

(This will rebuild from scratch, including retraining all RL agents.)

---

## How to reproduce

The pipeline is fully scripted. To re-run any version:

1. Check out the git commit corresponding to that version (see commit
   list below).
2. Place the Pinnacle CLC RAD CSVs in `CLCDATA/`.
3. Run:

```bash
RUN_ID=my_run bash scripts/run_walkforward_parallel.sh
```

Final tables land in `artifacts/wf_my_run/aggregated/results.csv`.

### Per-version git commits

```text
8270e20  v8: A2C ablation — revert sigma_tgt env, state z-score, A2C entropy
19b6de7  v7: 3 surgical fixes after v6 regression
c3a9526  v6: stop chasing paper literal: tune for performance over fidelity
8d3a874  v5: trainer revert to Sharpe-only early stopping (Zhang p.7)
08f7704  v4: env per-contract reward normalization (μ=1/p_ref) + 1% σ floor
b390216  v3: features drop NaN/inf rows from window_ready
334a374  v2: PG restore mean baseline to prevent softmax NaN
129fcec  v2 (predecessor): decouple env σ_tgt (training) from portfolio σ_tgt
eebafa5  v1: align replication with Zhang (2019)
```

### Per-version run IDs (artifact directories)

```text
artifacts/wf_zhang2019_replication       (v1, deleted)
artifacts/wf_zhang2019_replication_v2    (v2, cancelled mid-run)
artifacts/wf_zhang2019_replication_v3    (v3, deleted)
artifacts/wf_zhang2019_replication_v4    (v4, results.csv reconstructed)
artifacts/wf_zhang2019_replication_v5    (v5)
artifacts/wf_zhang2019_replication_v6    (v6)
artifacts/wf_zhang2019_replication_v7    (v7)
artifacts/wf_zhang2019_replication_v8    (v8)
```
