from backtesting.lib import FractionalBacktest
import pandas as pd
import numpy as np
from strategy import NuggetStrategy

# ── Configuration ──────────────────────────────────────────────
N_CHUNKS = 8          # number of time-based splits
CASH     = 1000
COMM     = 0.002
MARGIN   = 1.0

# ── Load data (DO NOT CHANGE) ─────────────────────────────────
df = pd.read_parquet("./data/cache/normalized.parquet")
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# ── Split into roughly equal time chunks ───────────────────────
# Keep chunks as pandas DataFrames, not NumPy arrays
chunk_indices = np.array_split(np.arange(len(df)), N_CHUNKS)
chunks = [df.iloc[idx].copy() for idx in chunk_indices if len(idx) > 0]

# ── Run backtest on each chunk ─────────────────────────────────
all_stats  = []
all_trades = []

for i, chunk in enumerate(chunks):
    if len(chunk) < 2:
        print(f"[Chunk {i+1}] skipped – too few rows")
        continue

    bt = FractionalBacktest(
        chunk,
        NuggetStrategy,
        cash=CASH,
        commission=COMM,
        margin=MARGIN,
        trade_on_close=True,
        finalize_trades=True,
    )
    stats = bt.run()

    # Collect key metrics
    row = {
        "chunk":          i + 1,
        "start":          chunk.index[0],
        "end":            chunk.index[-1],
        "rows":           len(chunk),
        "return_pct":     stats["Return [%]"],
        "sharpe":         stats.get("Sharpe Ratio", np.nan),
        "max_drawdown":   stats.get("Max. Drawdown [%]", np.nan),
        "win_rate":       stats.get("Win Rate [%]", np.nan),
        "num_trades":     stats.get("# Trades", 0),
    }
    all_stats.append(row)

    trades = stats._trades.copy()
    if not trades.empty:
        trades["chunk"] = i + 1
        all_trades.append(trades)

    print(f"\n{'='*60}")
    print(f"  Chunk {i+1}  |  {row['start']}  →  {row['end']}  ({row['rows']} rows)")
    print(f"{'='*60}")
    print(stats.drop(['_strategy', '_equity_curve', '_trades']))

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
print("  CHUNK SUMMARY")
print(f"{'='*60}")
print(summary.to_string(index=False))

print(f"\n{'='*60}")
print("  CONSISTENCY CHECK")
print(f"{'='*60}")
print(f"  Chunks profitable : {(summary['return_pct'] > 0).sum()} / {len(summary)}")
print(f"  Avg return/chunk  : {summary['return_pct'].mean():.2f}%")
print(f"  Std return/chunk  : {summary['return_pct'].std():.2f}%")
print(f"  Avg Sharpe/chunk  : {summary['sharpe'].mean():.4f}")
print(f"  Avg win rate      : {summary['win_rate'].mean():.2f}%")
print(f"  Full-data return  : {stats_full['Return [%]']:.2f}%")
print(f"  Full-data Sharpe  : {stats_full.get('Sharpe Ratio', np.nan):.4f}")

# ── Overfit warning ────────────────────────────────────────────
profitable_chunks = (summary['return_pct'] > 0).sum()
if profitable_chunks < len(summary) * 0.5:
    print("\n  ⚠️  WARNING: Strategy is profitable in fewer than half the")
    print("     chunks — likely overfitted to a specific regime.")
elif summary['return_pct'].std() > abs(summary['return_pct'].mean()) * 2:
    print("\n  ⚠️  WARNING: High variance across chunks — performance may")
    print("     be regime-dependent rather than robust.")
else:
    print("\n  ✅  Strategy shows reasonable consistency across chunks.")