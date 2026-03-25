from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def rsi(values, n):
    s = pd.Series(values)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    rs = avg_up / avg_down
    out = pd.Series(100 - (100 / (1 + rs)))
    return out.fillna(50.0).to_numpy(copy=True)


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
    return np.asarray(pd.Series(tr).rolling(n).mean(), dtype=float)


class NuggetStrategy(Strategy):
    fast_n = 21
    slow_n = 100
    rsi_n = 14
    atr_n = 14

    pullback_pct = 0.0035
    min_vol = 0.0025
    max_vol = 0.05

    risk_per_trade = 0.004
    atr_stop = 2.4
    atr_target = 4.5
    atr_trail = 2.0

    cooldown_bars = 12
    max_hold_bars = 96

    def init(self):
        self.fast = self.I(lambda x: ema(x, self.fast_n), self.data.Close)
        self.slow = self.I(lambda x: ema(x, self.slow_n), self.data.Close)
        self.rsi = self.I(lambda x: rsi(x, self.rsi_n), self.data.Close)
        self.atr = self.I(
            lambda h, l, c: atr(h, l, c, self.atr_n),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

        self._entry_bar = -1
        self._sl_price = None
        self._tp_price = None
        self._cooldown = 0

    def next(self):
        if len(self.data.Close) < 2:
            return

        if self._cooldown > 0:
            self._cooldown -= 1

        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        fast = self.fast[-1]
        fast_prev = self.fast[-2]
        slow = self.slow[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr_now = self.atr[-1]

        if any(np.isnan(v) for v in [price, prev_price, fast, fast_prev, slow, rsi_now, rsi_prev, atr_now]):
            return

        atr_pct = atr_now / price if price > 0 else np.nan
        if np.isnan(atr_pct):
            return

        if self.position:
            self._manage_open_position(price, slow, rsi_now, atr_now)
            return

        if self._cooldown > 0:
            return

        if not (self.min_vol <= atr_pct <= self.max_vol):
            return

        uptrend = fast > slow and price > slow
        downtrend = fast < slow and price < slow

        long_pullback = prev_price < fast_prev * (1 - self.pullback_pct) and price > fast
        short_pullback = prev_price > fast_prev * (1 + self.pullback_pct) and price < fast

        long_momentum = rsi_prev < 42 and rsi_now > 48
        short_momentum = rsi_prev > 58 and rsi_now < 52

        size = self._size_fraction(atr_pct)
        if uptrend and (long_pullback or long_momentum):
            self._sl_price = price - self.atr_stop * atr_now
            self._tp_price = price + self.atr_target * atr_now
            self.buy(size=size, sl=self._sl_price, tp=self._tp_price)
            self._entry_bar = len(self.data.Close) - 1
            return

        if downtrend and (short_pullback or short_momentum):
            self._sl_price = price + self.atr_stop * atr_now
            self._tp_price = price - self.atr_target * atr_now
            self.sell(size=size, sl=self._sl_price, tp=self._tp_price)
            self._entry_bar = len(self.data.Close) - 1

    def _manage_open_position(self, price, slow, rsi_now, atr_now):
        bars_held = len(self.data.Close) - 1 - self._entry_bar

        if self.position.is_long:
            trail = price - self.atr_trail * atr_now
            self._sl_price = trail if self._sl_price is None else max(self._sl_price, trail)
            if (
                price <= self._sl_price
                or (self._tp_price is not None and price >= self._tp_price)
                or price < slow
                or rsi_now > 72
                or bars_held >= self.max_hold_bars
            ):
                self._close_position()
                return

        if self.position.is_short:
            trail = price + self.atr_trail * atr_now
            self._sl_price = trail if self._sl_price is None else min(self._sl_price, trail)
            if (
                price >= self._sl_price
                or (self._tp_price is not None and price <= self._tp_price)
                or price > slow
                or rsi_now < 28
                or bars_held >= self.max_hold_bars
            ):
                self._close_position()

    def _size_fraction(self, atr_pct):
        if atr_pct <= 0:
            return 0.02
        raw = self.risk_per_trade / (self.atr_stop * atr_pct)
        return float(min(max(raw, 0.02), 0.2))

    def _close_position(self):
        self.position.close()
        self._entry_bar = -1
        self._sl_price = None
        self._tp_price = None
        self._cooldown = self.cooldown_bars
