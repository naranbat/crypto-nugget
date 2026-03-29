# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto Nugget is an automated Bitcoin (BTCUSDT) trading strategy optimization framework. The workflow is an iteration loop: generate a new `strategy.py` → run walk-forward backtests → archive results → repeat.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run walk-forward backtest (primary command)
python simulate.py

# Rebuild data cache (only needed when updating raw data)
python prepare.py
```

## Architecture

**Fixed infrastructure (never modify):**
- `simulate.py` — Walk-forward backtesting engine. Loads `data/cache/normalized.parquet`, splits into 6-month in-sample / 1-month out-of-sample folds, runs `NuggetStrategy`, prints per-fold metrics and consistency checks.
- `prepare.py` — Aggregates raw Binance zip archives from `data/spot/BTCUSDT/` into the normalized parquet cache.

**The only file you modify:**
- `strategy.py` — Contains `NuggetStrategy(Strategy)` with `init()` (declare indicators) and `next()` (entry/exit logic). Helper functions (`ema`, `rsi`, `atr`, `supertrend_direction`) live here too.

**Archive:** `backup/YYYYMMDD-HHMMSS_<return>.py` — immutable snapshots of every strategy iteration.

## Iteration Loop

1. Run `python simulate.py` → record baseline metrics (return, trade count, Sharpe)
2. Write a completely new `strategy.py` from scratch (do NOT edit incrementally)
3. Run `python simulate.py` → extract metrics
4. Archive to `backup/YYYYMMDD-HHMMSS_<return>.py` (only if better than previous)
5. Repeat — run continuously until explicitly stopped

## Strategy Design

Each new strategy should meaningfully diverge from previous ones. Explore combinations of:
- **Indicators:** MAs, RSI, MACD, Bollinger Bands, ATR, Supertrend, etc.
- **Entry/exit logic:** breakout, mean-reversion, trend-following, momentum
- **Risk management:** position sizing, stop-loss, take-profit, drawdown limits
- **Filters:** trend, volatility, momentum

Prefer strategies with:
- Consistently positive walk-forward folds
- Acceptable worst out-of-sample drawdown
- Limited degradation under commission stress
- Higher weight given to recent folds when evaluating performance

## Key Constants (simulate.py)

```python
IS_MONTHS  = 6      # In-sample training window
OOS_MONTHS = 1      # Out-of-sample test step
CASH       = 1000   # Starting capital
COMM       = 0.002  # 0.2% commission
```

## Constraints

- Only `strategy.py` may be modified. `simulate.py` and `prepare.py` are immutable.
- Never use future data (no lookahead bias).
- Never overwrite existing backup files.
- After 8+ iterations without meaningful improvement, discard the current approach and start fresh with a new strategy paradigm.
