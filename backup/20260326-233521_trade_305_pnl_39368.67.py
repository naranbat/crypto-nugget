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


def rolling_mean(values, period):
    return pd.Series(values, dtype=float).rolling(period).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_period = 402
    breakout_period = 158
    atr_period = 25
    stop_atr = 5.77
    momentum_band = 0.0125
    atr_min_pct = 0.00115
    atr_max_pct = 0.0529
    size = 0.99999999999

    vol_period = 59
    vol_confirm = 1.45
    vol_cont = 0.95
    width_lookback = 209
    width_comp_mult = 0.744
    width_abs_max = 0.109

    def init(self):
        self.trend = self.I(lambda x: np.array(ema_np(x, self.trend_period), copy=True), self.data.Close)
        self.range_high = self.I(lambda x: np.array(rolling_prev_high(x, self.breakout_period), copy=True), self.data.High)
        self.range_low = self.I(lambda x: np.array(rolling_prev_low(x, self.breakout_period), copy=True), self.data.Low)
        self.atr = self.I(
            lambda h, l, c: np.array(atr_wilder(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.vol_ema = self.I(lambda v: np.array(ema_np(v, self.vol_period), copy=True), self.data.Volume)
        self.channel_width = self.I(
            lambda h, l, c: np.array(((pd.Series(h) - pd.Series(l)) / pd.Series(c)).to_numpy(copy=True), copy=True),
            self.range_high,
            self.range_low,
            self.data.Close,
        )
        self.width_avg = self.I(lambda w: np.array(rolling_mean(w, self.width_lookback), copy=True), self.channel_width)
        self.stop_price = np.nan

    def next(self):
        price = float(self.data.Close[-1])
        trend = float(self.trend[-1])
        breakout_high = float(self.range_high[-1])
        breakout_low = float(self.range_low[-1])
        atr_now = float(self.atr[-1])
        volume = float(self.data.Volume[-1])
        vol_ema = float(self.vol_ema[-1])
        width_now = float(self.channel_width[-1])
        width_avg = float(self.width_avg[-1])

        if any(np.isnan(v) for v in (price, trend, breakout_high, breakout_low, atr_now, volume, vol_ema, width_now, width_avg)):
            return

        if price <= 0 or trend <= 0 or vol_ema <= 0:
            return

        atr_pct = atr_now / price
        if atr_pct < self.atr_min_pct or atr_pct > self.atr_max_pct:
            if self.position:
                self.position.close()
            return

        momentum = (price / trend) - 1.0
        vol_ratio = volume / vol_ema
        compressed = width_now < self.width_abs_max and width_now < width_avg * self.width_comp_mult
        volume_gate = vol_ratio > self.vol_confirm if compressed else vol_ratio > self.vol_cont

        long_signal = price > breakout_high and price > trend and momentum > self.momentum_band and volume_gate
        short_signal = price < breakout_low and price < trend and momentum < -self.momentum_band and volume_gate

        if not self.position:
            if long_signal:
                self.buy(size=self.size)
                self.stop_price = price - self.stop_atr * atr_now
            elif short_signal:
                self.sell(size=self.size)
                self.stop_price = price + self.stop_atr * atr_now
            return

        if self.position.is_long:
            self.stop_price = max(self.stop_price, price - self.stop_atr * atr_now)
            if price <= self.stop_price or price < trend:
                self.position.close()
                if short_signal:
                    self.sell(size=self.size)
                    self.stop_price = price + self.stop_atr * atr_now
            return

        if self.position.is_short:
            self.stop_price = min(self.stop_price, price + self.stop_atr * atr_now)
            if price >= self.stop_price or price > trend:
                self.position.close()
                if long_signal:
                    self.buy(size=self.size)
                    self.stop_price = price - self.stop_atr * atr_now
