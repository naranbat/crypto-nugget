from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def rolling_high(values, n):
    return pd.Series(values).rolling(n).max().shift(1).to_numpy(copy=True)


def rolling_low(values, n):
    return pd.Series(values).rolling(n).min().shift(1).to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_n = 150
    breakout_n = 40
    exit_n = 15
    atr_n = 20
    size = 0.5
    atr_mult = 3.0

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_n), copy=True), self.data.Close)
        self.breakout = self.I(lambda x: np.array(rolling_high(x, self.breakout_n), copy=True), self.data.High)
        self.exit_line = self.I(lambda x: np.array(rolling_low(x, self.exit_n), copy=True), self.data.Low)
        self.atr = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

    def next(self):
        price = self.data.Close[-1]
        trend = self.trend[-1]
        breakout = self.breakout[-1]
        exit_line = self.exit_line[-1]
        atr_now = self.atr[-1]

        if np.isnan(trend) or np.isnan(breakout) or np.isnan(exit_line) or np.isnan(atr_now):
            return

        if not self.position:
            if price > trend and price > breakout:
                sl = price - self.atr_mult * atr_now
                self.buy(size=self.size, sl=sl)
        elif price < exit_line or price < trend:
            self.position.close()
