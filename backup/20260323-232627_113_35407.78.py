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
    trend_n = 70
    breakout_n = 10
    exit_n = 11
    atr_n = 12
    size = 0.99
    atr_mult = 1.0
    min_vol = 0.01
    max_vol = 0.16

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_n), copy=True), self.data.Close)
        self.high = self.I(lambda x: np.array(rolling_high(x, self.breakout_n), copy=True), self.data.High)
        self.low = self.I(lambda x: np.array(rolling_low(x, self.breakout_n), copy=True), self.data.Low)
        self.exit_low = self.I(lambda x: np.array(rolling_low(x, self.exit_n), copy=True), self.data.Low)
        self.exit_high = self.I(lambda x: np.array(rolling_high(x, self.exit_n), copy=True), self.data.High)
        self.atr = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

    def next(self):
        price = self.data.Close[-1]
        trend = self.trend[-1]
        high = self.high[-1]
        low = self.low[-1]
        exit_low = self.exit_low[-1]
        exit_high = self.exit_high[-1]
        atr_now = self.atr[-1]

        if np.isnan(trend) or np.isnan(high) or np.isnan(low) or np.isnan(exit_low) or np.isnan(exit_high) or np.isnan(atr_now):
            return

        atr_pct = atr_now / price if price > 0 else 0.0
        if atr_pct < self.min_vol or atr_pct > self.max_vol:
            if self.position:
                self.position.close()
            return

        if not self.position:
            if price > trend and price > high:
                self.buy(size=self.size, sl=price - self.atr_mult * atr_now)
            elif price < trend and price < low:
                self.sell(size=self.size, sl=price + self.atr_mult * atr_now)
            return

        if self.position.is_long and (price < exit_low or price < trend):
            self.position.close()
        elif self.position.is_short and (price > exit_high or price > trend):
            self.position.close()
