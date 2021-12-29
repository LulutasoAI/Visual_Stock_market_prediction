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

class estimation():
    def __init__(self):
        try:
            self.stock_id = sys.argv[1]
        except:
            print("input stock symbol or something ex) ^N225")
            sys.exit()
        General_util = PictureProcessing()
        Transfer_learning_class = Transfer_learning()

        General_util.create_folder_if_None_exists("Yogensho")
        General_util.create_folder_if_None_exists("temp_image_for_estimation")
        General_util.create_folder_if_None_exists("weights")
        General_util.create_folder_if_None_exists(os.path.join("weights","best"))
        General_util.create_folder_if_None_exists(os.path.join("weights","last"))

        weightpath = os.path.join("weights","best")
        weightpathlast = os.path.join("weights","last")
        self.the_latest_model_name = os.path.join(weightpath,"weights.hdf5")
        self.the_latest_model_name_last = os.path.join(weightpathlast,"latest_model.hdf5")
        #the best model
        self.model = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name)
        #the last model
        self.model2 = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name_last)
        self.temp_image_path = os.path.join("temp_image_for_estimation","image.png")
        self.path_logs = "Yogensho"


    def main(self):
        image_to_predict = self.Create_Image()
        #result_best = np.argmax(self.model.predict(image_to_predict))
        result_best = self.model.predict(image_to_predict)[0][0]
        result_last = self.model2.predict(image_to_predict)[0][0]
        
        print(result_best,"best  :  last",result_last)
        print("model 1 says {} percentage up".format(round((result_best*100),1)))
        print("model 2 says {} percentage up".format(round((result_last*100),1)))
        summary_result_string = "Model 1 says {}%_up and Model 2 says {}%_up ".format(str(round((result_best*100),1)),str(round((result_last*100),1)))
        self.logger(image_to_predict,summary_result_string)
        """
        result_last = np.argmax(self.model2.predict(image_to_predict))
        print(result_best,"best  :  last",result_last)
        if result_best == 0 and result_last==0:
            print("up")
        elif (result_best == 0 and result_last == 1) or (result_best == 1 and result_last == 0):
            print("neutral")
        elif result_best == 1 and result_last == 1:
            print("down")
        else:
            print("is this possible? ", result_best, result_last)
        """

    def get_date(self):
        d = datetime.datetime.now().strftime('%Y%m%d%H%M')
        return d

    def logger(self,image,summary_result_string):
        plt.imshow(image.reshape(256,256,3))
        plt.xticks([])
        plt.yticks([])
        plt.title("{} in {}, the last price was {}".format(self.stock_id,self.date,self.last_price))
        plt.xlabel(summary_result_string)
        plt.savefig(os.path.join(self.path_logs,"{}_{}.png".format(self.stock_id,self.date)))

    def Create_Image(self):
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
        df = df[[ "Open", "High", "Low", "Close", "Volume"]]
        df = df[-40:]
        if len(df)==40:
            print("okay")
        else:
            sys.exit()
        df_ = df.copy()
        mpf.plot(df_,type="candle", mav=(3,8,21),style='yahoo',closefig=True,savefig=self.temp_image_path)  
        image = Image.open(self.temp_image_path)
        #image.show()
        w, h = image.size
        box = (w*0.2,h*0.13,w*0.89,h*0.8)
        image_to_predict = np.array(image.crop(box).convert('RGB').resize((256,256))).reshape(1,256,256,3)
        return image_to_predict



if __name__ == "__main__":
    estimation().main()