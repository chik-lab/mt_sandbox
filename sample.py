import os
import sys

import pandas as pd

# Add the algo directory to Python path
algo_path = os.path.join("/mnt/e/code/algo")
sys.path.insert(0, algo_path)
from data.data_extractor import SmartDataExtractor

extractor = SmartDataExtractor()
from minitrade.backtest import Strategy


class MyStrategy(Strategy):
    def init(self):
        self.roc = self.I(self.data.ta.roc(10), name="ROC")  # calculate

    def next(self):
        self.alloc.assume_zero()  # initialize
        roc = self.roc.df.iloc[-1]  # get
        (
            self.alloc.bucket["equity"]
            .append(roc.sort_values(ascending=False), roc > 0)  # create
            .trim(3)  # add
            .weight_explicitly(1 / 3)  # keep
            .apply()
        )  # all
        self.rebalance(cash_reserve=0.01)  # apply


from minitrade.backtest import Backtest

stockList = ["MMM", "AXP", "AAPL", "BA"]
start_date = "2019-07-07"
end_date = "2021-10-07"
interval = "1d"
data = {}
for symbol in stockList:
    data[symbol] = extractor.extract_data_smart(symbol, start_date, end_date, interval)
df = pd.concat(data, axis=1).sort_index().ffill()
bt = Backtest(df, MyStrategy, cash=1000)
bt.run()
bt.plot(plot_allocation=True)
