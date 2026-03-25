from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest


@dataclass
class Params:
    fast_n: int
    slow_n: int
    atr_n: int
    band: float
    min_vol: float
    max_vol: float
    size: float
    max_hold: int


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return np.asarray(tr.rolling(n).mean(), dtype=float)


def make_strategy(params: Params):
    class WalkForwardStrategy(Strategy):
        fast_n = params.fast_n
        slow_n = params.slow_n
        atr_n = params.atr_n
        band = params.band
        min_vol = params.min_vol
        max_vol = params.max_vol
        size = params.size
        max_hold = params.max_hold

        def init(self):
            self.fast = self.I(lambda x: np.array(ema(x, self.fast_n), copy=True), self.data.Close)
            self.slow = self.I(lambda x: np.array(ema(x, self.slow_n), copy=True), self.data.Close)
            self.atr = self.I(
                lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
                self.data.High,
                self.data.Low,
                self.data.Close,
            )
            self._entry_bar = -1

        def next(self):
            price = self.data.Close[-1]
            fast = self.fast[-1]
            slow = self.slow[-1]
            atr_now = self.atr[-1]
            if np.isnan(price) or np.isnan(fast) or np.isnan(slow) or np.isnan(atr_now):
                return

            atr_pct = atr_now / price if price > 0 else 0.0
            if atr_pct < self.min_vol or atr_pct > self.max_vol:
                if self.position:
                    self.position.close()
                return

            go_long = fast > slow * (1 + self.band)
            go_short = fast < slow * (1 - self.band)

            if not self.position:
                if go_long:
                    self.buy(size=self.size)
                    self._entry_bar = len(self.data.Close) - 1
                elif go_short:
                    self.sell(size=self.size)
                    self._entry_bar = len(self.data.Close) - 1
                return

            if len(self.data.Close) - 1 - self._entry_bar >= self.max_hold:
                self.position.close()
                return

            if self.position.is_long and go_short:
                self.position.close()
                self.sell(size=self.size)
                self._entry_bar = len(self.data.Close) - 1
            elif self.position.is_short and go_long:
                self.position.close()
                self.buy(size=self.size)
                self._entry_bar = len(self.data.Close) - 1

    return WalkForwardStrategy


def run_backtest(df: pd.DataFrame, params: Params) -> dict:
    bt = FractionalBacktest(
        df,
        make_strategy(params),
        cash=1000,
        commission=0.002,
        margin=1.0,
        trade_on_close=True,
        finalize_trades=True,
    )
    s = bt.run()

    def _to_float(value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    def _to_int(value) -> int:
        return int(np.asarray(value).reshape(-1)[0])

    return {
        "pnl": _to_float(s["Equity Final [$]"] - 1000),
        "return_pct": _to_float(s["Return [%]"]),
        "trades": _to_int(s["# Trades"]),
        "sharpe": _to_float(s["Sharpe Ratio"]),
        "calmar": _to_float(s["Calmar Ratio"]),
        "max_dd": _to_float(s["Max. Drawdown [%]"]),
        "win_rate": _to_float(s["Win Rate [%]"]),
    }


def random_params(rng: np.random.Generator, anchor: Params | None = None) -> Params:
    if anchor is None or rng.random() < 0.2:
        fast_n = int(rng.integers(5, 55))
        slow_n = int(rng.integers(fast_n + 1, 120))
        atr_n = int(rng.integers(10, 100))
        band = float(rng.uniform(0.0005, 0.01))
        min_vol = float(rng.uniform(0.0, 0.01))
        max_vol = float(rng.uniform(0.08, 0.35))
        size = float(rng.uniform(0.35, 0.99))
        max_hold = int(rng.integers(240, 2400))
    else:
        step = 1.0 if rng.random() < 0.75 else 2.0
        fast_n = int(np.clip(round(anchor.fast_n + rng.normal(0, 2.8 * step)), 3, 80))
        slow_n = int(np.clip(round(anchor.slow_n + rng.normal(0, 4.0 * step)), 5, 140))
        atr_n = int(np.clip(round(anchor.atr_n + rng.normal(0, 8.0 * step)), 8, 140))
        band = float(np.clip(anchor.band + rng.normal(0, 0.0010 * step), 0.0002, 0.02))
        min_vol = float(np.clip(anchor.min_vol + rng.normal(0, 0.0012 * step), 0.0, 0.03))
        max_vol = float(np.clip(anchor.max_vol + rng.normal(0, 0.02 * step), 0.08, 0.45))
        size = float(np.clip(anchor.size + rng.normal(0, 0.04 * step), 0.25, 0.99))
        max_hold = int(np.clip(round(anchor.max_hold + rng.normal(0, 180 * step)), 120, 4000))

    if slow_n <= fast_n:
        slow_n = fast_n + int(rng.integers(1, 24))
    if max_vol <= min_vol:
        max_vol = min_vol + 0.02

    return Params(
        fast_n=fast_n,
        slow_n=slow_n,
        atr_n=atr_n,
        band=band,
        min_vol=min_vol,
        max_vol=max_vol,
        size=size,
        max_hold=max_hold,
    )


def objective(metrics: dict) -> float:
    inactivity_penalty = max(0, 8 - metrics["trades"]) * 500.0
    return metrics["pnl"] - 170.0 * abs(metrics["max_dd"]) - inactivity_penalty


def optimize_on_train(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    trials: int,
    anchor: Params,
) -> tuple[Params, dict]:
    best_p = anchor
    best_m = run_backtest(train_df, best_p)
    best_score = objective(best_m)

    for _ in range(trials):
        p = random_params(rng, best_p)
        try:
            m = run_backtest(train_df, p)
        except Exception:
            continue
        score = objective(m)
        if score > best_score:
            best_p, best_m, best_score = p, m, score

    return best_p, best_m


def build_folds(index: pd.Index, train_years: int = 3, test_months: int = 6) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    dt_index = pd.DatetimeIndex(index)
    start = pd.Timestamp(str(dt_index.min())).floor("D")
    end = pd.Timestamp(str(dt_index.max())).floor("D")
    cur = start

    while True:
        train_start = cur
        train_end = cur + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append((train_start, train_end, train_end, test_end))
        cur = cur + pd.DateOffset(months=test_months)

    return folds


def main():
    df = pd.read_parquet("./data/cache/normalized.parquet")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    idx = pd.DatetimeIndex(df.index)

    base = Params(
        fast_n=17,
        slow_n=20,
        atr_n=53,
        band=0.0034739190382566377,
        min_vol=0.0019457069229671042,
        max_vol=0.21170291407994768,
        size=0.9446983937981164,
        max_hold=954,
    )

    folds = build_folds(idx, train_years=3, test_months=6)
    rng = np.random.default_rng(20260325)

    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        train = df.loc[(idx >= tr_s) & (idx < tr_e)].copy()
        test = df.loc[(idx >= te_s) & (idx < te_e)].copy()
        if len(train) < 2000 or len(test) < 500:
            continue

        best_p, train_m = optimize_on_train(train, rng, trials=120, anchor=base)
        test_m = run_backtest(test, best_p)
        baseline_test_m = run_backtest(test, base)

        rows.append(
            {
                "fold": i,
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "test_start": str(te_s.date()),
                "test_end": str(te_e.date()),
                "train_pnl": train_m["pnl"],
                "train_sharpe": train_m["sharpe"],
                "train_max_dd": train_m["max_dd"],
                "test_pnl": test_m["pnl"],
                "test_sharpe": test_m["sharpe"],
                "test_max_dd": test_m["max_dd"],
                "test_trades": test_m["trades"],
                "baseline_test_pnl": baseline_test_m["pnl"],
                "baseline_test_sharpe": baseline_test_m["sharpe"],
                "baseline_test_max_dd": baseline_test_m["max_dd"],
                "param_fast_n": best_p.fast_n,
                "param_slow_n": best_p.slow_n,
                "param_atr_n": best_p.atr_n,
                "param_band": best_p.band,
                "param_min_vol": best_p.min_vol,
                "param_max_vol": best_p.max_vol,
                "param_size": best_p.size,
                "param_max_hold": best_p.max_hold,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No folds were generated; check date index or fold settings.")
        return

    def _series_float(series_obj) -> float:
        return float(np.asarray(series_obj).reshape(-1)[0])

    summary = {
        "folds": int(len(out_df)),
        "oos_total_pnl": _series_float(out_df[["test_pnl"]].sum()),
        "oos_median_pnl": _series_float(out_df[["test_pnl"]].median()),
        "oos_positive_fold_ratio": _series_float((out_df["test_pnl"] > 0).mean()),
        "oos_mean_sharpe": _series_float(out_df[["test_sharpe"]].mean()),
        "oos_worst_dd": _series_float(out_df[["test_max_dd"]].min()),
        "baseline_oos_total_pnl": _series_float(out_df[["baseline_test_pnl"]].sum()),
        "baseline_oos_positive_fold_ratio": _series_float((out_df["baseline_test_pnl"] > 0).mean()),
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"walk_forward_{stamp}.csv"
    json_path = reports_dir / f"walk_forward_{stamp}.json"

    out_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("WALK_FORWARD_SUMMARY", summary)
    print("WALK_FORWARD_CSV", str(csv_path))
    print("WALK_FORWARD_JSON", str(json_path))


if __name__ == "__main__":
    main()
