import pandas as pd
import numpy as np
import pandas_datareader as dt
import datetime
from matplotlib import pyplot as plt
from typing import Tuple


class Baseutils():
    """
    utils
    """
    def __init__(self,start=None,end=None):
        self.start, self.end = self.variables_init(start, end)

        
    def variables_init(self,start,end) -> Tuple[str, str]:
        if start == None:
            start = datetime.datetime(2015,4,1)
        if end == None:
            end = datetime.datetime.today()
        return start, end

    def fetch_prices(self,symbol:str) -> np.array:
        dataframe = dt.DataReader(symbol,"yahoo", self.start, self.end)
        prices = dataframe["Close"]
        return np.array(prices)

    def fetch_DataFrame(self,symbol:str) -> pd.DataFrame:
        dataframe = dt.DataReader(symbol,"yahoo", self.start, self.end)
        return dataframe