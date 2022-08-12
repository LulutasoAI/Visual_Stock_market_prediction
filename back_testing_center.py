from turtle import position
import pandas as pd 
import numpy as np
from utilities import Baseutils
import random
from typing import List, Tuple
from matplotlib import pyplot as plt
from Decisions import Position_generotor

class Back_test:
    def __init__(self, symbol:str):
        self.baseutils = Baseutils() #if you want to specify start and end date put them here.
        self.dataframe:pd.DataFrame = self.baseutils.fetch_DataFrame(symbol)
        self.prices : pd.DataFrame = self.dataframe["Close"]
        self.length = len(self.prices)
    
    def make_returns(self) -> pd.DataFrame:
        prices :pd.Series = self.prices
        Returns : pd.Series  = np.log(prices/prices.shift(1))
        return Returns

    def make_random_positions(self) -> List[int]:
        """
        Replace this with a real model
        """
        strategy = [] 
        for i in range(len(self.prices)):
            strategy.append(random.randint(-1,1))
        return np.array(strategy)
    
    def create_data_frame_for_calc_result(self, returns:List[float], positions:List[int] ) -> pd.DataFrame:
        data :pd.DataFrame = pd.DataFrame()
        if len(returns) != len(positions):
            print("Something is apparently wrong.")
            print("returnslength : ", len(returns)," positions_length: ",len(positions))
            return 
        else:
            data["returns"] = returns
            data["positions"] = positions
            return data

    def make_result(self,data:pd.DataFrame):
        last = 100
        data["profit"] = data["returns"].shift(1)*data["positions"]
        plt.plot(data["returns"][-last:])
        plt.plot(data["profit"][-last:])
        plt.show()
        original_capital = 10000 
        result_profit = original_capital * np.exp(np.sum(data["profit"]))
        hold_profit = original_capital * np.exp(np.sum(data["returns"]))
        print(int(result_profit-original_capital),"result_profit with strat")
        print(int(hold_profit-original_capital),"theoretical profit when holding until the last day in data.")
