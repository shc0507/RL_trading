# Remaining Issues

## 1. Rolling 5-Year Retraining
Currently: train once on 2005-2015, test on 2019+.
Paper: retrain every 5 years using all data up to that point, then test on the next 5 years.
Why it matters: single split can be lucky or unlucky. Walk-forward is more rigorous.

## 2. ETFs vs. Futures Data
Currently: using ETFs (SPY, GLD, etc.) via yfinance.
Paper: uses 50 back-adjusted continuous futures from the CLC Database.
Why it matters: commodity ETFs have contango drag (USO/UNG lose 20-30%/yr from roll costs). All ETFs have expense ratios baked into prices. Results won't match the paper's numbers.
Options: yfinance `=F` tickers (free, has roll jumps), Norgate Data (~$30/mo, proper back-adjusted), CLC/Pinnacle ($149 one-time).

## 3. Cross-Asset Training
Currently: one model per ticker.
Paper: one model per asset class (e.g., one commodity model trained on all commodities).
Why it matters: more training data, better generalization. Not a bug, just a design choice.

## 4. Sharpe for Early Stopping
Currently: early stopping based on validation Sharpe ratio.
Paper: doesn't specify. Sharpe is reasonable but may favor flat low-variance strategies.
Low priority — current approach is defensible.
