from backtesting import Strategy
import numpy as np
import pandas as pd


def ema_np(values, period):
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().to_numpy(copy=True)


def atr_wilder(high, low, close, period):
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev = c.shift(1)
    tr = pd.concat(((h - l).abs(), (h - prev).abs(), (l - prev).abs()), axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy(copy=True)


def rolling_prev_high(values, period):
    return pd.Series(values, dtype=float).rolling(period).max().shift(1).to_numpy(copy=True)


def rolling_prev_low(values, period):
    return pd.Series(values, dtype=float).rolling(period).min().shift(1).to_numpy(copy=True)


def momentum_roc(values, period):
    series = pd.Series(values, dtype=float)
    return ((series / series.shift(period)) - 1.0).to_numpy(copy=True)


class NuggetStrategy(Strategy):
    fast_period = 16
    slow_period = 81
    trend_period = 265
    breakout_period = 53
    roc_period = 21
    atr_period = 7

    long_roc_min = 0.030910146526815552
    short_roc_max = 0.002975183102719159

    min_atr_pct = 0.003039753862515585
    max_atr_pct = 0.0960871850666967

    stop_atr = 3.0095292196835226
    take_atr = 11.32951745050069
    trail_atr = 2.663241600062334

    max_hold = 106
    size = 0.9423033522082253

    def init(self):
        self.fast = self.I(lambda x: np.array(ema_np(x, self.fast_period), copy=True), self.data.Close)
        self.slow = self.I(lambda x: np.array(ema_np(x, self.slow_period), copy=True), self.data.Close)
        self.trend = self.I(lambda x: np.array(ema_np(x, self.trend_period), copy=True), self.data.Close)
        self.roc = self.I(lambda x: np.array(momentum_roc(x, self.roc_period), copy=True), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: np.array(atr_wilder(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.range_high = self.I(
            lambda x: np.array(rolling_prev_high(x, self.breakout_period), copy=True),
            self.data.High,
        )
        self.range_low = self.I(
            lambda x: np.array(rolling_prev_low(x, self.breakout_period), copy=True),
            self.data.Low,
        )
        self.entry_bar = -1
        self.stop_price = np.nan
        self.take_price = np.nan

    def _clear_state(self):
        self.entry_bar = -1
        self.stop_price = np.nan
        self.take_price = np.nan

    def _enter_long(self, price, atr_now, bar_idx):
        self.buy(size=self.size)
        self.entry_bar = bar_idx
        self.stop_price = price - self.stop_atr * atr_now
        self.take_price = price + self.take_atr * atr_now

    def _enter_short(self, price, atr_now, bar_idx):
        self.sell(size=self.size)
        self.entry_bar = bar_idx
        self.stop_price = price + self.stop_atr * atr_now
        self.take_price = price - self.take_atr * atr_now

    def next(self):
        price = float(self.data.Close[-1])
        fast = float(self.fast[-1])
        slow = float(self.slow[-1])
        trend = float(self.trend[-1])
        roc_now = float(self.roc[-1])
        atr_now = float(self.atr[-1])
        breakout_high = float(self.range_high[-1])
        breakout_low = float(self.range_low[-1])

        if any(np.isnan(v) for v in (price, fast, slow, trend, roc_now, atr_now, breakout_high, breakout_low)):
            return

        atr_pct = atr_now / price if price > 0 else 0.0
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            if self.position:
                self.position.close()
                self._clear_state()
            return

        bar_idx = len(self.data.Close) - 1
        long_signal = price > breakout_high and fast > slow and price > trend and roc_now >= self.long_roc_min
        short_signal = price < breakout_low and fast < slow and price < trend and roc_now <= self.short_roc_max

        if not self.position:
            if long_signal:
                self._enter_long(price, atr_now, bar_idx)
            elif short_signal:
                self._enter_short(price, atr_now, bar_idx)
            return

        timeout = self.entry_bar >= 0 and (bar_idx - self.entry_bar) >= self.max_hold

        if self.position.is_long:
            self.stop_price = max(self.stop_price, price - self.trail_atr * atr_now)
            if price <= self.stop_price or price >= self.take_price or timeout or fast < slow or price < trend:
                self.position.close()
                self._clear_state()
                if short_signal:
                    self._enter_short(price, atr_now, bar_idx)
                return

        if self.position.is_short:
            self.stop_price = min(self.stop_price, price + self.trail_atr * atr_now)
            if price >= self.stop_price or price <= self.take_price or timeout or fast > slow or price > trend:
                self.position.close()
                self._clear_state()
                if long_signal:
                    self._enter_long(price, atr_now, bar_idx)
