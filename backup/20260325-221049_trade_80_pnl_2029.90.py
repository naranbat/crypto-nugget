from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, period):
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().to_numpy(copy=True)


def atr(high, low, close, period):
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy(copy=True)


def highest(values, period):
    return pd.Series(values, dtype=float).rolling(period).max().shift(1).to_numpy(copy=True)


def lowest(values, period):
    return pd.Series(values, dtype=float).rolling(period).min().shift(1).to_numpy(copy=True)


def roc(values, period):
    s = pd.Series(values, dtype=float)
    return ((s / s.shift(period)) - 1.0).to_numpy(copy=True)


class NuggetStrategy(Strategy):
    fast_period = 12
    slow_period = 34
    trend_period = 220
    breakout_period = 30
    roc_period = 6
    atr_period = 28

    roc_long_min = 0.015
    roc_short_max = -0.015

    min_atr_pct = 0.0012
    max_atr_pct = 0.1

    stop_atr = 2.8
    take_atr = 3.0
    trail_atr = 1.6

    max_hold = 180
    size = 0.95

    def init(self):
        self.fast = self.I(lambda x: np.array(ema(x, self.fast_period), copy=True), self.data.Close)
        self.slow = self.I(lambda x: np.array(ema(x, self.slow_period), copy=True), self.data.Close)
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_period), copy=True), self.data.Close)
        self.roc_val = self.I(lambda x: np.array(roc(x, self.roc_period), copy=True), self.data.Close)
        self.atr_val = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.hh = self.I(lambda x: np.array(highest(x, self.breakout_period), copy=True), self.data.High)
        self.ll = self.I(lambda x: np.array(lowest(x, self.breakout_period), copy=True), self.data.Low)
        self._entry_bar = -1
        self._stop_price = np.nan
        self._take_price = np.nan

    def _open_long(self, price, atr_now, bar_index):
        self.buy(size=self.size)
        self._entry_bar = bar_index
        self._stop_price = price - self.stop_atr * atr_now
        self._take_price = price + self.take_atr * atr_now

    def _open_short(self, price, atr_now, bar_index):
        self.sell(size=self.size)
        self._entry_bar = bar_index
        self._stop_price = price + self.stop_atr * atr_now
        self._take_price = price - self.take_atr * atr_now

    def _reset_risk(self):
        self._entry_bar = -1
        self._stop_price = np.nan
        self._take_price = np.nan

    def next(self):
        price = float(self.data.Close[-1])
        fast = float(self.fast[-1])
        slow = float(self.slow[-1])
        trend = float(self.trend[-1])
        roc_now = float(self.roc_val[-1])
        atr_now = float(self.atr_val[-1])
        hh = float(self.hh[-1])
        ll = float(self.ll[-1])

        if any(np.isnan(v) for v in [price, fast, slow, trend, roc_now, atr_now, hh, ll]):
            return

        atr_pct = atr_now / price if price > 0 else 0.0
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            if self.position:
                self.position.close()
                self._reset_risk()
            return

        bar_index = len(self.data.Close) - 1

        long_signal = price > hh and fast > slow and price > trend and roc_now >= self.roc_long_min
        short_signal = price < ll and fast < slow and price < trend and roc_now <= self.roc_short_max

        if not self.position:
            if long_signal:
                self._open_long(price, atr_now, bar_index)
            elif short_signal:
                self._open_short(price, atr_now, bar_index)
            return

        timeout = self._entry_bar >= 0 and (bar_index - self._entry_bar) >= self.max_hold

        if self.position.is_long:
            trailing_stop = price - self.trail_atr * atr_now
            self._stop_price = max(self._stop_price, trailing_stop)
            stop_hit = price <= self._stop_price
            take_hit = price >= self._take_price
            invalidated = fast < slow or price < trend

            if stop_hit or take_hit or timeout or invalidated:
                self.position.close()
                self._reset_risk()
                if short_signal:
                    self._open_short(price, atr_now, bar_index)
                return

        if self.position.is_short:
            trailing_stop = price + self.trail_atr * atr_now
            self._stop_price = min(self._stop_price, trailing_stop)
            stop_hit = price >= self._stop_price
            take_hit = price <= self._take_price
            invalidated = fast > slow or price > trend

            if stop_hit or take_hit or timeout or invalidated:
                self.position.close()
                self._reset_risk()
                if long_signal:
                    self._open_long(price, atr_now, bar_index)
