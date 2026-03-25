# Crypto Nugget

# Objective
Continuously iterate and improve `strategy.py` by running simulations and archiving every attempt with its performance metrics.

---

# Baseline Step
1. Run:
   python simulate.py
2. Record the latest output metrics (at minimum: trade count and PnL).

---

# Iteration Loop (Repeat indefinitely until explicitly stopped)

For each iteration:

1. Strategy Creation
   - Generate a completely new version of `strategy.py` from scratch.
   - Do NOT reuse or edit previous code incrementally.
   - Do NOT modify `simulate.py`.

2. Simulation
   - Run:
     python simulate.py

3. Result Extraction
   - Parse the output and extract:
     - Trade count
     - PnL

4. Archival
   - Ensure `backup/` directory exists (create if missing).
   - Save the exact `strategy.py` used in this iteration.
   - Filename format:
     YYYYMMDD-HHMMSS_<trade_count>_<pnl>.py
   - Example:
     20260323-141530_77_332.02.py

5. Logging (optional but recommended)
   - Print or log:
     - Timestamp
     - Trade count
     - PnL
     - Short description of strategy idea

---

# Backup Rules
- Archive EVERY attempt (including worse-performing ones).
- Never overwrite existing files.
- Maintain full history of all strategies tested.

---

# Strategy Design Guidelines
Each new strategy should explore meaningful variation using:

1. Indicators
   - e.g., moving averages, RSI, MACD, Bollinger Bands

2. Entry/Exit Logic
   - Refine conditions for trade signals
   - Avoid trivial or duplicate logic

3. Risk Management
   - Position sizing
   - Stop loss / take profit
   - Max drawdown constraints

4. Filters (optional but encouraged)
   - Trend filters
   - Volatility filters
   - Momentum filters

5. Exploration Principle
   - Prefer diversity over minor tweaks
   - Avoid repeating similar strategies
   - Occasionally try unconventional combinations

---

# Constraints
- Only `strategy.py` may be modified.
- `simulate.py` must remain unchanged.
- Read backup files if needed.
- The loop must run continuously until a stop instruction is given.
- Do NOT cheat or using forward-looking data

---

# Robust Validation (`simulate_eval.py`)

Use `simulate_eval.py` for out-of-sample checks and stability diagnostics.

## Common Commands
- Run all checks:
  `python simulate_eval.py`
- Holdout only:
  `python simulate_eval.py --mode holdout --holdout-months 24`
- Walk-forward only:
  `python simulate_eval.py --mode walkforward --train-years 3 --test-months 6`
- Yearly only:
  `python simulate_eval.py --mode yearly`
- Cost/execution sensitivity only:
  `python simulate_eval.py --mode sensitivity`

## What The Output Means
`simulate_eval.py` writes files to `reports/`:
- `simulate_eval_<timestamp>.json` (high-level summary)
- `simulate_eval_<timestamp>.csv` (row-level details per split/window)

Interpret key summary fields as follows:
- `full_sample.pnl`: in-sample PnL on the full dataset (useful context, not enough for robustness on its own).
- `holdout.test_pnl`: PnL on the final unseen holdout window.
- `holdout.test_train_pnl_ratio`: quick overfit signal; very low values mean train performance does not carry into holdout.
- `walkforward.oos_total_pnl`: total out-of-sample PnL over all walk-forward test folds.
- `walkforward.oos_positive_fold_ratio`: fraction of test folds with positive PnL; higher is generally better.
- `walkforward.oos_mean_sharpe`: average Sharpe across test folds.
- `walkforward.oos_worst_dd`: worst drawdown among test folds (more negative = riskier).
- `yearly.positive_year_ratio`: fraction of positive years.
- `sensitivity.commission_0p2_pnl` vs `sensitivity.commission_0p6_pnl`: how much performance degrades when costs increase.

## Practical Rule
Prefer strategies with:
- Consistently positive walk-forward folds,
- Acceptable worst out-of-sample drawdown,
- Reasonable holdout performance,
- And limited degradation under higher commission.
