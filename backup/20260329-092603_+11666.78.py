from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, period):
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().to_numpy(copy=True)


def rsi(values, period):
    close = pd.Series(values, dtype=float)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1.0 / period, adjust=False).mean() / down.ewm(alpha=1.0 / period, adjust=False).mean()
    return (100.0 - 100.0 / (1.0 + rs)).to_numpy(copy=True)


def atr(high, low, close, period):
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev_close = c.shift(1)
    tr = pd.concat(((h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()), axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(copy=True)


def supertrend_direction(high, low, close, period, multiplier):
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    atr_now = atr(h, l, c, period)

    hl2 = (h + l) / 2.0
    upper_band = hl2 + multiplier * atr_now
    lower_band = hl2 - multiplier * atr_now

    final_upper = np.full_like(c, np.nan)
    final_lower = np.full_like(c, np.nan)
    supertrend = np.full_like(c, np.nan)
    direction = np.zeros(len(c), dtype=float)

    for i in range(len(c)):
        if i == 0 or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            direction[i] = 0.0
            continue

        prev_upper = final_upper[i - 1] if not np.isnan(final_upper[i - 1]) else upper_band[i - 1]
        prev_lower = final_lower[i - 1] if not np.isnan(final_lower[i - 1]) else lower_band[i - 1]
        prev_close = c[i - 1]

        final_upper[i] = upper_band[i] if (upper_band[i] < prev_upper or prev_close > prev_upper) else prev_upper
        final_lower[i] = lower_band[i] if (lower_band[i] > prev_lower or prev_close < prev_lower) else prev_lower

        prev_supertrend = supertrend[i - 1]
        if np.isnan(prev_supertrend):
            if c[i] >= final_upper[i]:
                supertrend[i] = final_lower[i]
                direction[i] = 1.0
            else:
                supertrend[i] = final_upper[i]
                direction[i] = -1.0
            continue

        if prev_supertrend == prev_upper:
            if c[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1.0
            else:
                supertrend[i] = final_lower[i]
                direction[i] = 1.0
        else:
            if c[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                direction[i] = 1.0
            else:
                supertrend[i] = final_upper[i]
                direction[i] = -1.0

    return direction


class NuggetStrategy(Strategy):
    # Winning base: st_period=20, st_mult=6.33, ema=83, buf=0.0116, rsi_period=20, rsi_long=64, rsi_short=35.6
    # Trying: wider ST stop (mult=7.5) to ride big bull moves longer
    st_period = 19
    st_multiplier = 6.65
    ema_period = 66
    ema_buffer = 0.011599252524932551
    rsi_period = 20
    rsi_long = 64.04350789001414
    rsi_short = 33.5
    size = 0.999999

    def init(self):
        self.st_dir = self.I(
            lambda h, l, c: np.array(supertrend_direction(h, l, c, self.st_period, self.st_multiplier), copy=True),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )
        self.ema_trend = self.I(lambda c: np.array(ema(c, self.ema_period), copy=True), self.data.Close)
        self.rsi_now = self.I(lambda c: np.array(rsi(c, self.rsi_period), copy=True), self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        direction = float(self.st_dir[-1])
        ema_now = float(self.ema_trend[-1])
        rsi_now = float(self.rsi_now[-1])

        if any(np.isnan(v) for v in (price, direction, ema_now, rsi_now)):
            return
        if price <= 0.0 or ema_now <= 0.0:
            return

        long_signal = (
            direction > 0.0
            and price > ema_now * (1.0 + self.ema_buffer)
            and rsi_now > self.rsi_long
        )
        short_signal = (
            direction < 0.0
            and price < ema_now * (1.0 - self.ema_buffer)
            and rsi_now < self.rsi_short
        )

        if long_signal:
            if not self.position or not self.position.is_long:
                if self.position:
                    self.position.close()
                self.buy(size=self.size)
            return

        if short_signal and (not self.position or not self.position.is_short):
            if self.position:
                self.position.close()
            self.sell(size=self.size)
