from backtesting import Strategy
import numpy as np
import pandas as pd


def sma(values, n):
    return pd.Series(values).rolling(n).mean().to_numpy(copy=True)


def std(values, n):
    return pd.Series(values).rolling(n).std(ddof=0).to_numpy(copy=True)


def rsi(values, n):
    series = pd.Series(values)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    rs = avg_up / avg_down
    return (100 - 100 / (1 + rs)).to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    bb_n = 20
    bb_k = 2.2
    trend_n = 150
    rsi_n = 3
    atr_n = 14
    entry_rsi = 12
    exit_rsi = 65
    size = 0.25
    sl_atr = 3.0

    def init(self):
        self.mid = self.I(lambda x: np.array(sma(x, self.bb_n), copy=True), self.data.Close)
        self.vol = self.I(lambda x: np.array(std(x, self.bb_n), copy=True), self.data.Close)
        self.trend = self.I(lambda x: np.array(sma(x, self.trend_n), copy=True), self.data.Close)
        self.rsi = self.I(lambda x: np.array(rsi(x, self.rsi_n), copy=True), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

    def next(self):
        price = self.data.Close[-1]
        mid = self.mid[-1]
        vol = self.vol[-1]
        trend = self.trend[-1]
        rsi_now = self.rsi[-1]
        atr_now = self.atr[-1]

        if np.isnan(mid) or np.isnan(vol) or np.isnan(trend) or np.isnan(rsi_now) or np.isnan(atr_now):
            return

        lower = mid - self.bb_k * vol
        if not self.position:
            if price > trend and price < lower and rsi_now < self.entry_rsi:
                sl = price - self.sl_atr * atr_now
                self.buy(size=self.size, sl=sl)
        elif price >= mid or rsi_now > self.exit_rsi:
            self.position.close()
