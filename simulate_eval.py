import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from strategy import NuggetStrategy


def scalar_float(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def scalar_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def run_backtest(df: pd.DataFrame, commission: float = 0.002, trade_on_close: bool = True) -> dict:
    bt = FractionalBacktest(
        df,
        NuggetStrategy,
        cash=1000,
        commission=commission,
        margin=1.0,
        trade_on_close=trade_on_close,
        finalize_trades=True,
    )
    stats = bt.run()
    return {
        "start": str(stats["Start"]),
        "end": str(stats["End"]),
        "bars": int(len(df)),
        "equity_final": scalar_float(stats["Equity Final [$]"]),
        "pnl": scalar_float(stats["Equity Final [$]"] - 1000),
        "return_pct": scalar_float(stats["Return [%]"]),
        "trades": scalar_int(stats["# Trades"]),
        "win_rate": scalar_float(stats["Win Rate [%]"]),
        "sharpe": scalar_float(stats["Sharpe Ratio"]),
        "calmar": scalar_float(stats["Calmar Ratio"]),
        "max_dd": scalar_float(stats["Max. Drawdown [%]"]),
    }


def load_data() -> pd.DataFrame:
    df = pd.read_parquet("./data/cache/normalized.parquet")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def validate_split(train: pd.DataFrame, test: pd.DataFrame, min_train_bars: int, min_test_bars: int):
    if len(train) < min_train_bars:
        raise ValueError(f"Train bars too small: {len(train)} < {min_train_bars}")
    if len(test) < min_test_bars:
        raise ValueError(f"Test bars too small: {len(test)} < {min_test_bars}")
    if train.index.max() >= test.index.min():
        raise ValueError("Train/test overlap detected")


def eval_holdout(df: pd.DataFrame, holdout_months: int, min_train_bars: int, min_test_bars: int) -> tuple[list[dict], dict]:
    idx = pd.DatetimeIndex(df.index)
    end = pd.Timestamp(str(idx.max())).floor("D")
    holdout_start = end - pd.DateOffset(months=holdout_months)

    train = df.loc[idx < holdout_start].copy()
    test = df.loc[idx >= holdout_start].copy()
    validate_split(train, test, min_train_bars, min_test_bars)

    train_m = run_backtest(train)
    test_m = run_backtest(test)

    rows = [
        {
            "section": "holdout",
            "split": "train",
            "start": str(train.index.min()),
            "end": str(train.index.max()),
            **train_m,
        },
        {
            "section": "holdout",
            "split": "test",
            "start": str(test.index.min()),
            "end": str(test.index.max()),
            **test_m,
        },
    ]
    summary = {
        "holdout_months": int(holdout_months),
        "train_pnl": train_m["pnl"],
        "test_pnl": test_m["pnl"],
        "test_train_pnl_ratio": (test_m["pnl"] / train_m["pnl"]) if train_m["pnl"] != 0 else None,
        "test_sharpe": test_m["sharpe"],
        "test_max_dd": test_m["max_dd"],
        "test_trades": test_m["trades"],
    }
    return rows, summary


def eval_walkforward(
    df: pd.DataFrame,
    train_years: int,
    test_months: int,
    min_train_bars: int,
    min_test_bars: int,
) -> tuple[list[dict], dict]:
    idx = pd.DatetimeIndex(df.index)
    start = pd.Timestamp(str(idx.min())).floor("D")
    end = pd.Timestamp(str(idx.max())).floor("D")
    cur = start

    rows = []
    fold = 0
    while True:
        train_start = cur
        train_end = cur + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        train = df.loc[(idx >= train_start) & (idx < train_end)].copy()
        test = df.loc[(idx >= train_end) & (idx < test_end)].copy()

        try:
            validate_split(train, test, min_train_bars, min_test_bars)
        except ValueError:
            cur = cur + pd.DateOffset(months=test_months)
            continue

        fold += 1
        m = run_backtest(test)
        rows.append(
            {
                "section": "walkforward",
                "fold": fold,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(train_end.date()),
                "test_end": str(test_end.date()),
                **m,
            }
        )
        cur = cur + pd.DateOffset(months=test_months)

    if not rows:
        raise ValueError("No valid walk-forward folds were produced")

    wf = pd.DataFrame(rows)
    summary = {
        "folds": int(len(wf)),
        "oos_total_pnl": scalar_float(wf[["pnl"]].sum()),
        "oos_median_pnl": scalar_float(wf[["pnl"]].median()),
        "oos_positive_fold_ratio": scalar_float((wf["pnl"] > 0).mean()),
        "oos_mean_sharpe": scalar_float(wf[["sharpe"]].mean()),
        "oos_worst_dd": scalar_float(wf[["max_dd"]].min()),
        "oos_total_trades": scalar_int(wf[["trades"]].sum()),
        "train_years": int(train_years),
        "test_months": int(test_months),
    }
    return rows, summary


def eval_yearly(df: pd.DataFrame, min_test_bars: int) -> tuple[list[dict], dict]:
    idx = pd.DatetimeIndex(df.index)
    rows = []
    for year in sorted(idx.year.unique()):
        seg = df.loc[idx.year == year].copy()
        if len(seg) < min_test_bars:
            continue
        m = run_backtest(seg)
        rows.append(
            {
                "section": "yearly",
                "year": int(year),
                **m,
            }
        )

    if not rows:
        raise ValueError("No valid yearly segments were produced")

    y = pd.DataFrame(rows)
    summary = {
        "years": int(len(y)),
        "positive_year_ratio": scalar_float((y["pnl"] > 0).mean()),
        "total_yearly_pnl": scalar_float(y[["pnl"]].sum()),
        "worst_year_pnl": scalar_float(y[["pnl"]].min()),
    }
    return rows, summary


def eval_sensitivity(df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows = []

    for c in [0.001, 0.002, 0.003, 0.004, 0.006]:
        m = run_backtest(df, commission=c, trade_on_close=True)
        rows.append(
            {
                "section": "sensitivity",
                "kind": "commission",
                "value": c,
                **m,
            }
        )

    for toc in [True, False]:
        m = run_backtest(df, commission=0.002, trade_on_close=toc)
        rows.append(
            {
                "section": "sensitivity",
                "kind": "trade_on_close",
                "value": toc,
                **m,
            }
        )

    sens_df = pd.DataFrame(rows)
    summary = {
        "rows": int(len(sens_df)),
        "commission_0p2_pnl": float(sens_df[(sens_df["kind"] == "commission") & (sens_df["value"] == 0.002)]["pnl"].iloc[0]),
        "commission_0p6_pnl": float(sens_df[(sens_df["kind"] == "commission") & (sens_df["value"] == 0.006)]["pnl"].iloc[0]),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Robust evaluation runner for NuggetStrategy")
    parser.add_argument("--mode", choices=["all", "holdout", "walkforward", "yearly", "sensitivity"], default="all")
    parser.add_argument("--holdout-months", type=int, default=24)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--min-train-bars", type=int, default=2000)
    parser.add_argument("--min-test-bars", type=int, default=500)
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    df = load_data()
    full = run_backtest(df)

    all_rows = [
        {
            "section": "full",
            "split": "full_sample",
            **full,
        }
    ]
    summary = {
        "full_sample": full,
    }

    if args.mode in {"all", "holdout"}:
        rows, s = eval_holdout(df, args.holdout_months, args.min_train_bars, args.min_test_bars)
        all_rows.extend(rows)
        summary["holdout"] = s

    if args.mode in {"all", "walkforward"}:
        rows, s = eval_walkforward(df, args.train_years, args.test_months, args.min_train_bars, args.min_test_bars)
        all_rows.extend(rows)
        summary["walkforward"] = s

    if args.mode in {"all", "yearly"}:
        rows, s = eval_yearly(df, args.min_test_bars)
        all_rows.extend(rows)
        summary["yearly"] = s

    if args.mode in {"all", "sensitivity"}:
        rows, s = eval_sensitivity(df)
        all_rows.extend(rows)
        summary["sensitivity"] = s

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = outdir / f"simulate_eval_{stamp}.csv"
    json_path = outdir / f"simulate_eval_{stamp}.json"

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("EVAL_FULL", full)
    if "holdout" in summary:
        print("EVAL_HOLDOUT", summary["holdout"])
    if "walkforward" in summary:
        print("EVAL_WALKFORWARD", summary["walkforward"])
    if "yearly" in summary:
        print("EVAL_YEARLY", summary["yearly"])
    if "sensitivity" in summary:
        print("EVAL_SENSITIVITY", summary["sensitivity"])
    print("EVAL_CSV", str(csv_path))
    print("EVAL_JSON", str(json_path))


if __name__ == "__main__":
    main()
