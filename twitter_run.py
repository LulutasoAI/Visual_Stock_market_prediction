from secret import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET

from requests_oauthlib import OAuth1Session
from http import HTTPStatus
from datetime import datetime



import tweepy
import datetime
import schedule
import time
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
from TransferLearning import Transfer_learning
import datetime 
from GeneralUtils import PictureProcessing
import random


class estimation():
    def __init__(self):
        stocks_favourite = ["GOOG","TSLA","AMZN", "NEE", "^N225","TM","AAPL"]
        print("")
        try:
            #self.stock_id = sys.argv[1]
            if sys.argv[1] == "1":
                randomizer = random.randint(0,len(stocks_favourite)-1)
                self.stock_id = stocks_favourite[randomizer]
            elif sys.argv[1] == "2":
                self.stock_id = sys.argv[2]
            elif sys.argv[1] == "3":
                #self.stock_id = "^N225"
                self.stock_id = "^DJI"
        except:
            print("input stock symbol or something ex) ^N225")
            print(self.stock_id)
            sys.exit()

        self.make_folders() #making necessary folders
        self.load_models() #load models and put them into self.model and so on.
        
        self.temp_image_path = os.path.join("temp_image_for_estimation","image.png")
        self.path_logs = "Yogensho"

    def main(self):
        image_to_predict = self.Create_Image()
        result_best = self.model.predict(image_to_predict)[0][0]
        result_last = self.model2.predict(image_to_predict)[0][0]
        
        print(result_best,"best  :  last",result_last)
        print("model 1 says {} percentage up".format(round((result_best*100),1)))
        print("model 2 says {} percentage up".format(round((result_last*100),1)))
        summary_result_string = "Model 1 says {}%_up and Model 2 says {}%_up ".format(str(round((result_best*100),1)),str(round((result_last*100),1)))
        self.logger(image_to_predict,summary_result_string)
        print("going to tweet")
        self.tweet(result_best,result_last)
        print("reached tweet")
        

    def make_folders(self):
        General_util = PictureProcessing()

        General_util.create_folder_if_None_exists("Yogensho")
        General_util.create_folder_if_None_exists("temp_image_for_estimation")
        General_util.create_folder_if_None_exists("weights")
        General_util.create_folder_if_None_exists(os.path.join("weights","best"))
        General_util.create_folder_if_None_exists(os.path.join("weights","last"))
        
        General_util.create_folder_if_None_exists("twitter")

    def load_models(self):
        Transfer_learning_class = Transfer_learning()
        
        weightpath = os.path.join("weights","best")
        weightpathlast = os.path.join("weights","last")
        self.the_latest_model_name = os.path.join(weightpath,"weights.hdf5")
        self.the_latest_model_name_last = os.path.join(weightpathlast,"latest_model.hdf5")
        #the best model
        self.model = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name)
        #the last model
        self.model2 = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name_last)

    def get_date(self):
        d = datetime.datetime.now().strftime('%Y%m%d%H%M')
        return d

    def tweet(self,result_best,result_last):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth ,wait_on_rate_limit = True)
        result_best = min(round((result_best*100),1),99.9) #最大99.9% because 100% is scammy
        result_last = min(round((result_last*100),1),99.9)
        average_result = round(np.mean([result_best,result_last]),1)
        if average_result >= 50.0:
            message = "${} {}%の確率で(40日後に)騰がる🖕🔥".format(self.stock_id,str(average_result))
        else:
            message = "${} {}%の確率で(40日後に)下がる👎🥶".format(self.stock_id,str(100.0-average_result))
        filename = os.path.join("twitter","result.png")
        message += "*Any prediction is bullshit. It is impossible to predict the market."
        api.update_status_with_media(filename=filename, status=message)

    def logger(self,image,summary_result_string):
        plt.imshow(image.reshape(256,256,3))
        plt.xticks([])
        plt.yticks([])
        plt.title("{} in {}, the last price was {}".format(self.stock_id,self.date,self.last_price))
        plt.xlabel(summary_result_string)
        plt.savefig(os.path.join(self.path_logs,"{}_{}.png".format(self.stock_id,self.date)))
        plt.savefig(os.path.join("twitter","result.png"))


    def Create_Image(self):
        df = self.fetch_data()
        image_to_predict = self.prices_to_Image(df)
        return image_to_predict

    def fetch_data(self):
        today = date.today()
        start_date = "2021-01-01" #Change it as you need.
        end_date = "{}".format(today)
        try:
            df = data.DataReader(self.stock_id, "yahoo", start_date, end_date)
            self.date = self.get_date() 
        except:
            print("unknown symbol")
            sys.exit()
        self.last_price = round(df["Close"][-1],2)
        print(self.last_price)
        return df

    def prices_to_Image(self,df):
        df = df[[ "Open", "High", "Low", "Close", "Volume"]]
        df = df[-40:]
        if len(df)==40:
            print("okay")
        else:
            sys.exit()
        df_ = df.copy()
        mpf.plot(df_,type="candle", mav=(3,8,21),style='yahoo',closefig=True,savefig=self.temp_image_path)  
        image = Image.open(self.temp_image_path)
        w, h = image.size
        box = (w*0.2,h*0.13,w*0.89,h*0.8)
        image_to_predict = np.array(image.crop(box).convert('RGB').resize((256,256))).reshape(1,256,256,3)
        return image_to_predict
        

if __name__ == "__main__":
    if sys.argv[1] == "1" or sys.argv[1] == "2" or sys.argv[1] == "3":
        estimation().main()
        sys.exit()
    def job():
        estimation().main()

    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("23:30").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)