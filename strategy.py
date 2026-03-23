from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np


def SMA(values, n):
    return pd.Series(values).rolling(n).mean().to_numpy(copy=True)

class NuggetStrategy(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.sma1 = self.I(lambda x: np.array(SMA(x, self.n1), copy=True), self.data.Close)
        self.sma2 = self.I(lambda x: np.array(SMA(x, self.n2), copy=True), self.data.Close)

    def next(self):
        if self.sma1[-1] > self.sma2[-1] and not self.position:
            self.buy(size=0.1)
        elif self.sma1[-1] < self.sma2[-1] and self.position:
            self.position.close()