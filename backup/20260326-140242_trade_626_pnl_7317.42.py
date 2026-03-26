from backtesting import Strategy
import numpy as np
import pandas as pd


def ema_np(values, period):
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    period = 800
    slope_lag = 36
    band = 0.002
    size = 0.99

    def init(self):
        self.regime_ema = self.I(lambda x: np.array(ema_np(x, self.period), copy=True), self.data.Close)

    def next(self):
        if len(self.regime_ema) <= self.slope_lag:
            return

        price = float(self.data.Close[-1])
        ema_now = float(self.regime_ema[-1])
        ema_prev = float(self.regime_ema[-1 - self.slope_lag])

        if any(np.isnan(v) for v in (price, ema_now, ema_prev)) or ema_now <= 0:
            return

        distance = (price / ema_now) - 1.0
        long_signal = distance > self.band and ema_now > ema_prev
        short_signal = distance < -self.band and ema_now < ema_prev

        if not self.position:
            if long_signal:
                self.buy(size=self.size)
            elif short_signal:
                self.sell(size=self.size)
            return

        if self.position.is_long and not long_signal:
            self.position.close()
            if short_signal:
                self.sell(size=self.size)
            return

        if self.position.is_short and not short_signal:
            self.position.close()
            if long_signal:
                self.buy(size=self.size)
