You are doing a code review of three files in a PyTorch project that recreates Zhang, Zohren, Roberts (2020) "Deep Reinforcement Learning for Trading" (JFDS). Do NOT edit files — review only.

**Files to review (read in full):**
- /Users/haochen/projects/RL_trading/rl_trading/config.py
- /Users/haochen/projects/RL_trading/rl_trading/data.py
- /Users/haochen/projects/RL_trading/rl_trading/features.py

**Read the paper directly to avoid relying on paraphrases.** The PDF is at:
/Users/haochen/projects/RL_trading/docs/Deep-Reinforcement-Learning-for-Trading (1).pdf

Focus on pages 4–5 (the "State space" section covers features: normalized close, vol-adjusted returns, MACD, RSI; the "Reward function" section uses the 60-day EWM vol). Use the Read tool with `pages: "4-5"` and skim pages 3 and 6–7 if useful for context.

**What I want you to check (think independently — flag anything else that looks wrong):**
1. Do feature formulas match the paper?
   - `norm_close`: paper says r_{t-252,t} / (σ_t · √252) — is the code's implementation correct?
   - Vol-adjusted returns over 1/2/3/12-month horizons (21/42/63/252 days) — correct?
   - MACD: paper defines q_t = (m(S) − m(L)) / std(p_{t-63:t}) and MACD_t = q_t / std(q_{t-252:t}), with scale pairs S∈{8,16,32}, L∈{24,48,96}. Check formula, rolling windows, EWMA vs SMA, and the combined signal (avg of φ(MACD) where φ(x) = x·exp(−x²/4)/0.89).
   - RSI: paper uses 30-day lookback; oscillates 0–100. Code scales to [0,1] via rs/(1+rs). Is that equivalent to RSI/100? Verify the math.
2. EWM volatility: paper specifies 60-day span on daily returns. Does the code match? Is `min_periods=span` reasonable? Annualization via √252 — consistent?
3. `window_ready` flag: does it gate enough history for all features (return_252, MACD's 63-day price std + 252-day q-std)?
4. `UNIVERSE` in config.py: paper uses 50 liquid futures (2011–2019). Repo uses ETFs. That's a known deviation (TODO.md acknowledges). Don't re-flag the deviation itself, but flag any issues in how the universe or splits are defined (split dates, leakage risk, missing validation of the universe list).
5. `data.py`: yfinance usage — any correctness issues? Cache key collisions? Handling of multi-index returns? `adj_close` fallback logic? Date tz-localize?
6. Any silent bugs: off-by-ones in shifts, lookahead bias, NaN propagation into features, EWM `adjust=True` default vs paper's intended formula.
7. Any dead code, confusing naming, or wasted computation (e.g., `ret_{h}` stored but only `ret_{h}_vol` used — is that intentional? Baseline `sign_r` references `ret_252`; check that it's consumed somewhere).

**Report format (under ~400 words):**
- One short paragraph with your overall take.
- A bulleted list of findings. For each: `file.py:line` reference, one-line problem statement, severity (bug / spec-deviation / nit). Don't pad with obvious praise.
- If you're unsure about something, say "UNSURE:" and explain.

Report back to me — the main Claude instance will consolidate findings across categories.
