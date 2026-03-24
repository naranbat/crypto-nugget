from backtesting import Strategy
import numpy as np
import pandas as pd


def ema(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def rolling_high(values, n):
    # No shift here — shifting is handled via I(...) offset or accepted as next-bar fill lag
    return pd.Series(values).rolling(n).max().to_numpy(copy=True)


def rolling_low(values, n):
    return pd.Series(values).rolling(n).min().to_numpy(copy=True)


def atr(high, low, close, n):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy(copy=True)


class NuggetStrategy(Strategy):
    trend_n = 70
    breakout_n = 18
    exit_n = 12
    atr_n = 20

    # ATR-based position sizing: risk X% of equity per trade
    risk_pct = 0.04
    atr_mult = 0.8          # SL = atr_mult * ATR from entry
    min_vol = 0.004
    max_vol = 0.08

    rebalance_every = 10
    rebalance_portion = 0.05
    rebalance_topup = 0.02
    
    def init(self):
        self.trend = self.I(lambda x: ema(x, self.trend_n), self.data.Close)

        self.high = self.I(
            lambda x: pd.Series(rolling_high(x, self.breakout_n)).shift(1).to_numpy(copy=True),
            self.data.High
        )
        self.low = self.I(
            lambda x: pd.Series(rolling_low(x, self.breakout_n)).shift(1).to_numpy(copy=True),
            self.data.Low
        )
        self.exit_low = self.I(
            lambda x: pd.Series(rolling_low(x, self.exit_n)).shift(1).to_numpy(copy=True),
            self.data.Low
        )
        self.exit_high = self.I(
            lambda x: pd.Series(rolling_high(x, self.exit_n)).shift(1).to_numpy(copy=True),
            self.data.High
        )
        self.atr = self.I(
            lambda h, l, c: atr(h, l, c, self.atr_n),
            self.data.High,
            self.data.Low,
            self.data.Close,
        )

        self._entry_price = None
        self._sl_price = None
        self._bars_in_trade = 0

    def next(self):
        price = self.data.Close[-1]
        trend = self.trend[-1]
        high = self.high[-1]
        low = self.low[-1]
        exit_low = self.exit_low[-1]
        exit_high = self.exit_high[-1]
        atr_now = self.atr[-1]

        if any(np.isnan(x) for x in [price, trend, high, low, exit_low, exit_high, atr_now]):
            return

        atr_pct = atr_now / price if price > 0 else np.nan
        if np.isnan(atr_pct):
            return

        # --- Volatility filter: only gates NEW entries, never forces an exit ---
        vol_ok = self.min_vol <= atr_pct <= self.max_vol

        # --- Manage existing position ---
        if self.position:
            self._bars_in_trade += 1

            if self.position.is_long:
                # Primary exits: channel low or trend
                if price < exit_low or price < trend:
                    self._close_position()
                    return
                # Hard SL guard (in case backtesting.py SL handling differs)
                if self._sl_price is not None and price <= self._sl_price:
                    self._close_position()
                    return

            elif self.position.is_short:
                if price > exit_high or price > trend:
                    self._close_position()
                    return
                if self._sl_price is not None and price >= self._sl_price:
                    self._close_position()
                    return

            # Periodic scale-out (aligned to bars IN trade, not global bar count)
            if self._bars_in_trade % self.rebalance_every == 0:
                self.position.close(portion=self.rebalance_portion)
                if self.rebalance_topup > 0:
                    if self.position.is_long:
                        self.buy(size=self.rebalance_topup)
                    elif self.position.is_short:
                        self.sell(size=self.rebalance_topup)

            return

        # --- No position: check for new entry ---
        if not vol_ok:
            return

        # ATR-based position sizing: size = (equity * risk_pct) / (atr_mult * atr_now)
        sl_distance = self.atr_mult * atr_now
        if sl_distance <= 0:
            return

        # Size as fraction of equity (units = equity_at_risk / (price * sl_distance_pct))
        equity = self.equity
        risk_amount = equity * self.risk_pct
        n_units = risk_amount / sl_distance
        size_fraction = (n_units * price) / equity
        size_fraction = min(max(size_fraction, 0.01), 0.95)  # clamp to [1%, 95%]

        if price > trend and price > high:
            sl = price - sl_distance
            self.buy(size=size_fraction, sl=sl)
            self._entry_price = price
            self._sl_price = sl
            self._bars_in_trade = 0

        elif price < trend and price < low:
            sl = price + sl_distance
            self.sell(size=size_fraction, sl=sl)
            self._entry_price = price
            self._sl_price = sl
            self._bars_in_trade = 0

    def _close_position(self):
        self.position.close()
        self._entry_price = None
        self._sl_price = None
        self._bars_in_trade = 0
