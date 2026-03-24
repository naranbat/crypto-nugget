from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy(copy=True)


def macd_hist(values):
    close = pd.Series(values)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return hist.to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_n = 180
    guard_n = 55
    atr_n = 14
    size = 0.35
    sl_atr = 2.2
    tp_atr = 6.0

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_n), copy=True), self.data.Close)
        self.guard = self.I(lambda x: np.array(ema(x, self.guard_n), copy=True), self.data.Close)
        self.hist = self.I(lambda x: np.array(macd_hist(x), copy=True), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

    def next(self):
        price = self.data.Close[-1]
        trend = self.trend[-1]
        guard = self.guard[-1]
        hist_now = self.hist[-1]
        hist_prev = self.hist[-2]
        atr_now = self.atr[-1]

        if np.isnan(trend) or np.isnan(guard) or np.isnan(hist_now) or np.isnan(hist_prev) or np.isnan(atr_now):
            return

        if not self.position:
            if price > trend and hist_prev <= 0 < hist_now:
                sl = price - self.sl_atr * atr_now
                tp = price + self.tp_atr * atr_now
                self.buy(size=self.size, sl=sl, tp=tp)
        elif hist_now < 0 or price < guard:
            self.position.close()
