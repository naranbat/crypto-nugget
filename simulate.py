from backtesting.lib import FractionalBacktest
import pandas as pd
from strategy import NuggetStrategy

# DO NOT CHANGE IT!
df = pd.read_parquet("./data/cache/normalized.parquet")
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
bt = FractionalBacktest(df, NuggetStrategy, cash=1000, commission=.002, margin=1.0)
stats = bt.run()
print(stats.drop(['_strategy', '_equity_curve', '_trades']))
