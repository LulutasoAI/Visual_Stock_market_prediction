from PIL import Image
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
from datetime import date
import stockidlst
import matplotlib.dates as mdates
import sys 
import mplfinance as mpf
from Utils import Utils
class Data_Generator():

    def __init__(self):
        self.stock_id_list = stockidlst.sidlist
        self.util = Utils()
        self.file = "Bestdata2010"
    
    def main(self):
        file = self.file
        cwd = os.getcwd() #current working directory?
        for first_check, stock_id in enumerate(self.stock_id_list):
            today = date.today()
            start_date = "2010-01-01" #Change it as you need.
            end_date = "{}".format(today)
            df = data.DataReader(stock_id, "yahoo", start_date, end_date)
            #df.index = pd.to_datetime(df.index)
            df = df[[ "Open", "High", "Low", "Close", "Volume"]]
            ct = 1
            start=1
            end = 40
            df_ = df.copy()
            stopper = 1
            if first_check == 0:  #creating reqired folders here. 
                self.util.create_folder_if_None_exists(file)
                self.util.create_folder_if_None_exists(os.path.join(file,"up"))
                self.util.create_folder_if_None_exists(os.path.join(file,"down"))
            while stopper == 1:
                try:
                    if (int(df["Close"][end+40]) - int(df["Close"][end])) / int(df["Close"][end]) *100 >= 5 :
                        df_ = df[start:end]
                        start += 40
                        end += 40
                        mpf.plot(df_,type="candle", mav=(3,8,21),style='yahoo',savefig=os.path.join(cwd, file, "up", "candlestick_data{}{}.png".format(stock_id,ct)))
                        ct += 1
                        print("up")
                    elif (int(df["Close"][end+40]) - int(df["Close"][end])) / int(df["Close"][end]) *100 <= -5 :
                        df_ = df[start:end]
                        start += 40
                        end += 40
                        mpf.plot(df_,type="candle", mav=(3,8,21),style='yahoo',savefig=os.path.join(cwd, file, "down", "candlestick_data{}{}.png".format(stock_id,ct)))
                        ct += 1
                        print("down")
                    else:
                        df_ = df[start:end]
                        start += 40
                        end += 40
                        ct += 1
                        print("ignored")
                except Exception as e:
                    print(e)
                    print("done")
                    print(stock_id)
                    break
                    stopper += 1
if __name__ == "__main__":
    cl = Data_Generator()
    cl.main()