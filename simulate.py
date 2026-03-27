from backtesting.lib import FractionalBacktest
import pandas as pd
import numpy as np
from strategy import NuggetStrategy

# ── Configuration ──────────────────────────────────────────────
IS_MONTHS  = 6       # in-sample window
OOS_MONTHS = 1       # out-of-sample step
CASH       = 1000
COMM       = 0.002
MARGIN     = 1.0

# ── Load data (DO NOT CHANGE) ──────────────────────────────────
df = pd.read_parquet("./data/cache/normalized.parquet")
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df.index = pd.to_datetime(df.index)

# ── Build walk-forward windows ─────────────────────────────────
def build_wf_windows(index, is_months, oos_months):
    """
    Rolls a fixed-length IS window forward by OOS_MONTHS each step.
    Yields (is_start, is_end, oos_start, oos_end) as Timestamps.
    """
    windows = []
    start = index[0]
    end   = index[-1]

    while True:
        is_start  = start
        is_end    = is_start  + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end   = oos_start + pd.DateOffset(months=oos_months)

        if oos_end > end:
            break

        windows.append((is_start, is_end, oos_start, oos_end))
        start = start + pd.DateOffset(months=oos_months)   # roll forward 1 month

    return windows

windows = build_wf_windows(df.index, IS_MONTHS, OOS_MONTHS)
print(f"Total walk-forward folds: {len(windows)}")

# ── Run walk-forward ───────────────────────────────────────────
all_stats  = []
all_trades = []

for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
    is_slice  = df[(df.index >= is_start)  & (df.index < is_end)].copy()
    oos_slice = df[(df.index >= oos_start) & (df.index < oos_end)].copy()

    if len(is_slice) < 2 or len(oos_slice) < 2:
        print(f"[Fold {i+1}] skipped – too few rows")
        continue

    # ── In-sample (fitting / optimisation would go here) ──────
    bt_is = FractionalBacktest(
        is_slice, NuggetStrategy,
        cash=CASH, commission=COMM, margin=MARGIN,
        trade_on_close=True, finalize_trades=True,
    )
    stats_is = bt_is.run()

    # ── Out-of-sample (evaluation) ─────────────────────────────
    bt_oos = FractionalBacktest(
        oos_slice, NuggetStrategy,
        cash=CASH, commission=COMM, margin=MARGIN,
        trade_on_close=True, finalize_trades=True,
    )
    stats_oos = bt_oos.run()

    row = {
        "fold":           i + 1,
        "is_start":       is_start.date(),
        "is_end":         is_end.date(),
        "oos_start":      oos_start.date(),
        "oos_end":        oos_end.date(),
        "is_rows":        len(is_slice),
        "oos_rows":       len(oos_slice),
        # in-sample metrics
        "is_return_pct":  stats_is["Return [%]"],
        "is_sharpe":      stats_is.get("Sharpe Ratio",      np.nan),
        "is_win_rate":    stats_is.get("Win Rate [%]",      np.nan),
        "is_trades":      stats_is.get("# Trades",          0),
        # out-of-sample metrics
        "oos_return_pct": stats_oos["Return [%]"],
        "oos_sharpe":     stats_oos.get("Sharpe Ratio",     np.nan),
        "oos_max_dd":     stats_oos.get("Max. Drawdown [%]",np.nan),
        "oos_win_rate":   stats_oos.get("Win Rate [%]",     np.nan),
        "oos_trades":     stats_oos.get("# Trades",         0),
    }
    all_stats.append(row)

    trades = stats_oos._trades.copy()
    if not trades.empty:
        trades["fold"] = i + 1
        all_trades.append(trades)

    print(f"\n{'='*60}")
    print(f"  Fold {i+1}")
    print(f"  IS : {is_start.date()} → {is_end.date()}  ({len(is_slice)} rows)")
    print(f"  OOS: {oos_start.date()} → {oos_end.date()}  ({len(oos_slice)} rows)")
    print(f"{'='*60}")
    print(f"  [IS]  Return: {row['is_return_pct']:.2f}%  |  Sharpe: {row['is_sharpe']:.4f}  |  Win Rate: {row['is_win_rate']:.2f}%  |  Trades: {row['is_trades']}")
    print(f"  [OOS] Return: {row['oos_return_pct']:.2f}%  |  Sharpe: {row['oos_sharpe']:.4f}  |  Win Rate: {row['oos_win_rate']:.2f}%  |  Trades: {row['oos_trades']}")

# ── Also run on the FULL dataset for comparison ────────────────
bt_full = FractionalBacktest(
    df, NuggetStrategy,
    cash=CASH, commission=COMM, margin=MARGIN,
    trade_on_close=True, finalize_trades=True,
)
stats_full = bt_full.run()

# ── Summary table ──────────────────────────────────────────────
summary = pd.DataFrame(all_stats)

print(f"\n{'='*60}")
print("  WALK-FORWARD SUMMARY  (OOS metrics)")
print(f"{'='*60}")
display_cols = ["fold", "oos_start", "oos_end", "oos_return_pct",
                "oos_sharpe", "oos_max_dd", "oos_win_rate", "oos_trades"]
print(summary[display_cols].to_string(index=False))

print(f"\n{'='*60}")
print("  CONSISTENCY CHECK  (OOS only)")
print(f"{'='*60}")
profitable = (summary["oos_return_pct"] > 0).sum()
total      = len(summary)
print(f"  Folds profitable  : {profitable} / {total}")
print(f"  Avg OOS return    : {summary['oos_return_pct'].mean():.2f}%")
print(f"  Std OOS return    : {summary['oos_return_pct'].std():.2f}%")
print(f"  Avg OOS Sharpe    : {summary['oos_sharpe'].mean():.4f}")
print(f"  Avg OOS win rate  : {summary['oos_win_rate'].mean():.2f}%")
print(f"  IS/OOS corr       : {summary['is_return_pct'].corr(summary['oos_return_pct']):.4f}")
print(f"  Full-data return  : {stats_full['Return [%]']:.2f}%")
print(f"  Full-data Sharpe  : {stats_full.get('Sharpe Ratio', np.nan):.4f}")

# ── Overfit / robustness warnings ─────────────────────────────
is_oos_corr = summary["is_return_pct"].corr(summary["oos_return_pct"])

if profitable < total * 0.5:
    print("\n  ⚠️  WARNING: Strategy is profitable in fewer than half the")
    print("     OOS folds — likely overfitted or regime-dependent.")
elif summary["oos_return_pct"].std() > abs(summary["oos_return_pct"].mean()) * 2:
    print("\n  ⚠️  WARNING: High OOS variance — performance may not be robust.")
elif is_oos_corr < 0.3:
    print("\n  ⚠️  WARNING: Low IS/OOS return correlation — in-sample results")
    print("     are a poor predictor of out-of-sample performance.")
else:
    print("\n  ✅  Strategy shows reasonable walk-forward consistency.")