from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return np.asarray(tr.rolling(n).mean(), dtype=float)


class NuggetStrategy(Strategy):
    fast_n = 19
    slow_n = 22
    atr_n = 45

    band = 0.0028804645484002294
    min_vol = 0.0
    max_vol = 0.18108717615883407

    size = 0.99
    max_hold = 827

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

        bars_held = len(self.data.Close) - 1 - self._entry_bar
        if bars_held >= self.max_hold:
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
