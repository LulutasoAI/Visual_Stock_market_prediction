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


#input prices. output positions.

class Position_generotor:
    def __init__(self) -> None:
        self.temp_image_path = os.path.join("temp_image_for_estimation","image.png")
        Transfer_learning_class = Transfer_learning()
        weightpath = os.path.join("weights","best")
        weightpathlast = os.path.join("weights","last")
        self.the_latest_model_name = os.path.join(weightpath,"weights.hdf5")
        self.the_latest_model_name_last = os.path.join(weightpathlast,"latest_model.hdf5")
        self.model = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name)
        self.model2 = Transfer_learning_class.load_Vgg16(load=True,model_path=self.the_latest_model_name_last)
        self.temp_image_path = os.path.join("temp_image_for_estimation","image.png")

    def make_positions(self, price_data:pd.DataFrame):
        price_data = price_data[[ "Open", "High", "Low", "Close", "Volume"]]
        positions = [0] *40
        for i in range(40,len(price_data)):
            current_data = price_data[i-40:i]
            if len(current_data)!=40:
                sys.exit()
            current_data = current_data.copy()
            mpf.plot(current_data ,type="candle", mav=(3,8,21),style='yahoo',closefig=True,savefig=self.temp_image_path)  
            image = Image.open(self.temp_image_path)
            #image.show()
            w, h = image.size
            box = (w*0.2,h*0.13,w*0.89,h*0.8)
            image_to_predict = np.array(image.crop(box).convert('RGB').resize((256,256))).reshape(1,256,256,3)
            positions.append(self.predict(image_to_predict))
        return positions

    
    def predict(self,image_to_predict):
        result_best = self.model.predict(image_to_predict)[0][0]
        result_last = self.model2.predict(image_to_predict)[0][0]
        if min(result_best, result_last)*100 > 50:
            return 1
        else:
            return -1
            



