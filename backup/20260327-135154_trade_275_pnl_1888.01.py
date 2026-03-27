from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, period):
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().to_numpy(copy=True)


def atr_wilder(high, low, close, period):
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev = c.shift(1)
    tr = pd.concat(((h - l).abs(), (h - prev).abs(), (l - prev).abs()), axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy(copy=True)


def highest_prev(values, period):
    return pd.Series(values, dtype=float).rolling(period).max().shift(1).to_numpy(copy=True)


def lowest_prev(values, period):
    return pd.Series(values, dtype=float).rolling(period).min().shift(1).to_numpy(copy=True)


def roc(values, period):
    s = pd.Series(values, dtype=float)
    return ((s / s.shift(period)) - 1.0).to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_period = 300
    entry_period = 140
    exit_period = 55
    atr_period = 22
    roc_period = 63
    vol_period = 48

    momentum_min = 0.055
    min_vol_ratio = 0.8
    atr_min_pct = 0.0010
    atr_max_pct = 0.058

    stop_atr = 4.8
    trail_atr = 4.0
    max_hold = 420
    size = 0.99999999

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_period), copy=True), self.data.Close)
        self.entry_high = self.I(lambda x: np.array(highest_prev(x, self.entry_period), copy=True), self.data.High)
        self.entry_low = self.I(lambda x: np.array(lowest_prev(x, self.entry_period), copy=True), self.data.Low)
        self.exit_high = self.I(lambda x: np.array(highest_prev(x, self.exit_period), copy=True), self.data.High)
        self.exit_low = self.I(lambda x: np.array(lowest_prev(x, self.exit_period), copy=True), self.data.Low)
        self.atr = self.I(
            lambda h, l, c: np.array(atr_wilder(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.mom = self.I(lambda x: np.array(roc(x, self.roc_period), copy=True), self.data.Close)
        self.vol_ema = self.I(lambda x: np.array(ema(x, self.vol_period), copy=True), self.data.Volume)
        self.entry_bar = -1
        self.stop_price = np.nan

    def _reset(self):
        self.entry_bar = -1
        self.stop_price = np.nan

    def _enter_long(self, price, atr_now, bar_index):
        self.buy(size=self.size)
        self.entry_bar = bar_index
        self.stop_price = price - self.stop_atr * atr_now

    def _enter_short(self, price, atr_now, bar_index):
        self.sell(size=self.size)
        self.entry_bar = bar_index
        self.stop_price = price + self.stop_atr * atr_now

    def next(self):
        price = float(self.data.Close[-1])
        trend = float(self.trend[-1])
        hh_entry = float(self.entry_high[-1])
        ll_entry = float(self.entry_low[-1])
        hh_exit = float(self.exit_high[-1])
        ll_exit = float(self.exit_low[-1])
        atr_now = float(self.atr[-1])
        mom = float(self.mom[-1])
        vol = float(self.data.Volume[-1])
        vol_ema = float(self.vol_ema[-1])

        if any(np.isnan(v) for v in (price, trend, hh_entry, ll_entry, hh_exit, ll_exit, atr_now, mom, vol, vol_ema)):
            return
        if price <= 0 or trend <= 0 or vol_ema <= 0:
            return

        atr_pct = atr_now / price
        if atr_pct < self.atr_min_pct or atr_pct > self.atr_max_pct:
            if self.position:
                self.position.close()
                self._reset()
            return

        vol_ratio = vol / vol_ema
        long_signal = price > hh_entry and price > trend and mom > self.momentum_min and vol_ratio > self.min_vol_ratio
        short_signal = price < ll_entry and price < trend and mom < -self.momentum_min and vol_ratio > self.min_vol_ratio

        bar_index = len(self.data.Close) - 1
        if not self.position:
            if long_signal:
                self._enter_long(price, atr_now, bar_index)
            elif short_signal:
                self._enter_short(price, atr_now, bar_index)
            return

        timeout = self.entry_bar >= 0 and (bar_index - self.entry_bar) >= self.max_hold

        if self.position.is_long:
            self.stop_price = max(self.stop_price, price - self.trail_atr * atr_now)
            close_long = price <= self.stop_price or price < ll_exit or price < trend or timeout
            if close_long:
                self.position.close()
                self._reset()
                if short_signal:
                    self._enter_short(price, atr_now, bar_index)
            return

        if self.position.is_short:
            self.stop_price = min(self.stop_price, price + self.trail_atr * atr_now)
            close_short = price >= self.stop_price or price > hh_exit or price > trend or timeout
            if close_short:
                self.position.close()
                self._reset()
                if long_signal:
                    self._enter_long(price, atr_now, bar_index)
