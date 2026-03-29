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
    tr = pd.concat(((h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()), axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(copy=True)


def rolling_high_prev(values, period):
    return pd.Series(values, dtype=float).rolling(period).max().shift(1).to_numpy(copy=True)


def rolling_low_prev(values, period):
    return pd.Series(values, dtype=float).rolling(period).min().shift(1).to_numpy(copy=True)


def rolling_mean(values, period):
    return pd.Series(values, dtype=float).rolling(period).mean().to_numpy(copy=True)


def rate_of_change(values, period):
    close = pd.Series(values, dtype=float)
    return (close / close.shift(period) - 1.0).to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_period = 399
    breakout_period = 164
    momentum_period = 68
    atr_period = 12

    momentum_gate = 0.0268
    stop_atr = 5.96
    lock_profit_atr = 2.27
    min_atr_pct = 0.00081
    max_atr_pct = 0.0491

    volume_period = 79
    vol_confirm = 2.28
    vol_continue = 1.08
    width_period = 93
    width_comp_mult = 0.771

    max_hold_bars = 479
    size = 0.99999999

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_period), copy=True), self.data.Close)
        self.break_high = self.I(lambda x: np.array(rolling_high_prev(x, self.breakout_period), copy=True), self.data.High)
        self.break_low = self.I(lambda x: np.array(rolling_low_prev(x, self.breakout_period), copy=True), self.data.Low)
        self.atr_now = self.I(
            lambda h, l, c: np.array(atr(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.momentum = self.I(lambda x: np.array(rate_of_change(x, self.momentum_period), copy=True), self.data.Close)
        self.volume_ema = self.I(lambda v: np.array(ema(v, self.volume_period), copy=True), self.data.Volume)
        self.channel_width = self.I(
            lambda h, l, c: np.array(((pd.Series(h) - pd.Series(l)) / pd.Series(c)).to_numpy(copy=True), copy=True),
            self.break_high,
            self.break_low,
            self.data.Close,
        )
        self.width_mean = self.I(lambda w: np.array(rolling_mean(w, self.width_period), copy=True), self.channel_width)

        self.stop_price = np.nan
        self.entry_price = np.nan
        self.hold_bars = 0

    def next(self):
        price = float(self.data.Close[-1])
        trend = float(self.trend[-1])
        breakout_high = float(self.break_high[-1])
        breakout_low = float(self.break_low[-1])
        atr_now = float(self.atr_now[-1])
        momentum = float(self.momentum[-1])
        volume = float(self.data.Volume[-1])
        volume_ema = float(self.volume_ema[-1])
        width = float(self.channel_width[-1])
        width_mean = float(self.width_mean[-1])

        if any(np.isnan(v) for v in (price, trend, breakout_high, breakout_low, atr_now, momentum, volume, volume_ema, width, width_mean)):
            return
        if price <= 0.0 or trend <= 0.0 or volume_ema <= 0.0:
            return

        atr_pct = atr_now / price
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            if self.position:
                self.position.close()
            self.hold_bars = 0
            return

        vol_ratio = volume / volume_ema
        compressed = width < width_mean * self.width_comp_mult
        volume_ok = vol_ratio > self.vol_confirm if compressed else vol_ratio > self.vol_continue

        long_signal = price > breakout_high and price > trend and momentum > self.momentum_gate and volume_ok
        short_signal = price < breakout_low and price < trend and momentum < -self.momentum_gate and volume_ok

        if not self.position:
            self.hold_bars = 0
            if long_signal:
                self.buy(size=self.size)
                self.entry_price = price
                self.stop_price = price - self.stop_atr * atr_now
            elif short_signal:
                self.sell(size=self.size)
                self.entry_price = price
                self.stop_price = price + self.stop_atr * atr_now
            return

        self.hold_bars += 1

        if self.position.is_long:
            self.stop_price = max(self.stop_price, price - self.stop_atr * atr_now)
            if price - self.entry_price > self.lock_profit_atr * atr_now:
                self.stop_price = max(self.stop_price, self.entry_price)

            if price <= self.stop_price or price < trend or self.hold_bars >= self.max_hold_bars:
                self.position.close()
                self.hold_bars = 0
                if short_signal:
                    self.sell(size=self.size)
                    self.entry_price = price
                    self.stop_price = price + self.stop_atr * atr_now
            return

        if self.position.is_short:
            self.stop_price = min(self.stop_price, price + self.stop_atr * atr_now)
            if self.entry_price - price > self.lock_profit_atr * atr_now:
                self.stop_price = min(self.stop_price, self.entry_price)

            if price >= self.stop_price or price > trend or self.hold_bars >= self.max_hold_bars:
                self.position.close()
                self.hold_bars = 0
                if long_signal:
                    self.buy(size=self.size)
                    self.entry_price = price
                    self.stop_price = price - self.stop_atr * atr_now
