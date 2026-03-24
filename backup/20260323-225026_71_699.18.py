from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def rsi(values, n=14):
    series = pd.Series(values)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.to_numpy(copy=True)


def atr(high, low, close, n=14):
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - close_s.shift(1)).abs(),
            (low_s - close_s.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    fast = 20
    slow = 80
    rsi_n = 14
    atr_n = 14
    rsi_long = 55
    rsi_exit = 45
    size = 0.2
    sl_atr = 2.5
    tp_atr = 5.0

    def init(self):
        self.ema_fast = self.I(lambda x: np.array(ema(x, self.fast), copy=True), self.data.Close)
        self.ema_slow = self.I(lambda x: np.array(ema(x, self.slow), copy=True), self.data.Close)
        self.rsi = self.I(lambda x: np.array(rsi(x, self.rsi_n), copy=True), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_n), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

    def next(self):
        price = self.data.Close[-1]
        atr_now = self.atr[-1]
        if np.isnan(self.ema_fast[-1]) or np.isnan(self.ema_slow[-1]) or np.isnan(self.rsi[-1]) or np.isnan(atr_now):
            return

        if not self.position:
            if self.ema_fast[-1] > self.ema_slow[-1] and self.rsi[-1] > self.rsi_long:
                sl = price - atr_now * self.sl_atr
                tp = price + atr_now * self.tp_atr
                if sl < price < tp:
                    self.buy(size=self.size, sl=sl, tp=tp)
        elif self.ema_fast[-1] < self.ema_slow[-1] or self.rsi[-1] < self.rsi_exit:
            self.position.close()
