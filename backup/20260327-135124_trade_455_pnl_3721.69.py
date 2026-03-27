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


def rolling_mean(values, period):
    return np.asarray(pd.Series(values, dtype=float).rolling(period).mean(), dtype=float)


def rsi(values, period):
    s = pd.Series(values, dtype=float)
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d.clip(upper=0))
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return np.asarray(out, dtype=float)


class NuggetStrategy(Strategy):
    trend_period = 423
    breakout_period = 68
    atr_period = 23
    vol_period = 33
    width_lookback = 156
    rsi_period = 13

    momentum_band = 0.0113
    stop_atr = 8.119
    trail_atr = 6.556
    breakeven_atr = 2.155
    max_hold = 356
    cooldown = 29

    atr_min_pct = 0.00147
    atr_max_pct = 0.096
    vol_confirm = 2.689
    vol_cont = 1.884
    width_comp_mult = 1.489
    width_abs_max = 0.205
    rsi_long_min = 74

    min_size = 0.2
    max_size = 0.98
    vol_target = 0.005

    def init(self):
        self.trend = self.I(lambda x: np.array(ema(x, self.trend_period), copy=True), self.data.Close)
        self.range_high = self.I(lambda x: np.array(highest_prev(x, self.breakout_period), copy=True), self.data.High)
        self.range_low = self.I(lambda x: np.array(lowest_prev(x, self.breakout_period), copy=True), self.data.Low)
        self.atr = self.I(
            lambda h, l, c: np.array(atr_wilder(h, l, c, self.atr_period), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.vol_ema = self.I(lambda x: np.array(ema(x, self.vol_period), copy=True), self.data.Volume)
        self.channel_width = self.I(
            lambda h, l, c: np.array(((pd.Series(h) - pd.Series(l)) / pd.Series(c)).to_numpy(copy=True), copy=True),
            self.range_high,
            self.range_low,
            self.data.Close,
        )
        self.width_avg = self.I(lambda x: np.array(rolling_mean(x, self.width_lookback), copy=True), self.channel_width)
        self.rsi = self.I(lambda x: np.array(rsi(x, self.rsi_period), copy=True), self.data.Close)

        self.stop_price = np.nan
        self.entry_price = np.nan
        self.entry_atr = np.nan
        self.entry_bar = -1
        self.last_exit_bar = -10**9

    def _size_from_vol(self, atr_pct):
        raw_size = self.vol_target / max(atr_pct, 1e-8)
        return float(min(self.max_size, max(self.min_size, raw_size)))

    def _reset(self, bar_index):
        self.stop_price = np.nan
        self.entry_price = np.nan
        self.entry_atr = np.nan
        self.entry_bar = -1
        self.last_exit_bar = bar_index

    def next(self):
        price = float(self.data.Close[-1])
        trend = float(self.trend[-1])
        breakout_high = float(self.range_high[-1])
        breakout_low = float(self.range_low[-1])
        atr_now = float(self.atr[-1])
        vol = float(self.data.Volume[-1])
        vol_ema = float(self.vol_ema[-1])
        width_now = float(self.channel_width[-1])
        width_avg = float(self.width_avg[-1])
        rsi_now = float(self.rsi[-1])
        bar_index = len(self.data.Close) - 1

        if any(np.isnan(v) for v in (price, trend, breakout_high, breakout_low, atr_now, vol, vol_ema, width_now, width_avg, rsi_now)):
            return
        if price <= 0 or trend <= 0 or vol_ema <= 0:
            return

        atr_pct = atr_now / price
        if atr_pct < self.atr_min_pct or atr_pct > self.atr_max_pct:
            if self.position:
                self.position.close()
                self._reset(bar_index)
            return

        momentum = (price / trend) - 1.0
        vol_ratio = vol / vol_ema
        compressed = width_now < self.width_abs_max and width_now < width_avg * self.width_comp_mult
        volume_gate = vol_ratio > self.vol_confirm if compressed else vol_ratio > self.vol_cont

        long_signal = (
            price > breakout_high
            and price > trend
            and momentum > self.momentum_band
            and volume_gate
            and rsi_now > self.rsi_long_min
        )

        if not self.position:
            if (bar_index - self.last_exit_bar) < self.cooldown:
                return

            if long_signal:
                self.buy(size=self._size_from_vol(atr_pct))
                self.stop_price = price - self.stop_atr * atr_now
                self.entry_price = price
                self.entry_atr = atr_now
                self.entry_bar = bar_index
            return

        if self.position.is_long:
            timeout = self.entry_bar >= 0 and (bar_index - self.entry_bar) >= self.max_hold

            if price - self.entry_price > self.breakeven_atr * self.entry_atr:
                self.stop_price = max(self.stop_price, self.entry_price)

            self.stop_price = max(self.stop_price, price - self.trail_atr * atr_now)

            if price <= self.stop_price or price < trend or timeout:
                self.position.close()
                self._reset(bar_index)
            return
