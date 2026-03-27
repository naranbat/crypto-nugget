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
     YYYYMMDD-HHMMSS_trade_<trade_count>_pnl_<pnl>.py
   - Example:
     20260323-141530_trade_77_pnl_332.02.py

5. Logging (optional but recommended)
   - Print or log:
     - Timestamp
     - Trade count
     - PnL
     - Short description of strategy idea

---

# Backup Rules
- Archive better result from previous one.
- Never overwrite existing files.
- Maintain full history of all strategies tested.

---

# Strategy Design Guidelines
Each new strategy should explore meaningful variation using:

0. Exploration Principle
   - Prefer diversity over minor tweaks
   - Avoid repeating similar strategies
   - Occasionally try unconventional combinations

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

---

# Constraints
- Only `strategy.py` may be modified.
- `simulate.py` must remain unchanged.
- Read backup files if needed.
- The loop must run continuously until a stop instruction is given.
- Do NOT cheat or using future data

---

# Robust Validation (`simulate.py`)
- Run all checks:
  `python simulate.py`
  
## Practical Rule
Prefer strategies with:
- Consistently positive walk-forward folds,
- Acceptable worst out-of-sample drawdown,
- Reasonable holdout performance,
- And limited degradation under higher commission.


## Final 
Iteratively optimize the trading strategy by adjusting its hyperparameters for at least 8 consecutive iterations. After each iteration, evaluate performance based on stability and profitability.

If no meaningful improvement is observed after these iterations, discard the current approach and initialize a new strategy. Repeat this process until a robust and stable strategy is achieved.

Primary objective:
- Achieve good profit and greater than previous one
- Maintain consistency and avoid overfitting
- Give higher weight to the most recent PnL when evaluating performance