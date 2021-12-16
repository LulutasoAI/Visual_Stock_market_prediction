import glob
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, Conv3D
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow.keras.layers import AlphaDropout, GaussianDropout, GaussianNoise
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
import datetime

class PictureProcessing():
    def __init__(self):
        """You can modify the _init_ process that can be set through config.ini"""
        self.picture_data_folder_name = "data2"
        self.folder_name_for_models = "newpickleddata"
        self.folder_name_for_backups = "backupdata"
        self.image_size = 256 #it could be 200, 50 or anything as you like.

    def main(self):
        """Define by yourself"""
        #X,Y = self.folder_name_to_X_and_Y()
        #self.XYpickler(X,Y)
        pass

    def get_currenttime_numeral(self):
        d = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return d
        
    def create_folder_if_None_exists(self,name):
        if not os.path.exists(name):
            os.makedirs(name)
            print("The folder named '{}' had not existed so I created it.".format(name))
        else:
            print("The folder named '{}' already exists so nothing was executed.".format(name))

    def XYpickler(self,X,Y):
        """
        1.Pickle X and Y as forms of X.sav and Y.sav. 2.Also create a new folder called backup if there is None.
        3.Then, save 'X_{}.sav'.format(datetimenoworsomething) and 'Y_{}.sav'.format(datetimenoworsomething) , as backups.
        """
        #procedure 1.
        #folder_name_for_models = "models"

        self.create_folder_if_None_exists(self.folder_name_for_models)
        filenameX = (os.path.join(self.folder_name_for_models,"X.sav"))
        pickle.dump(X, open(filenameX, "wb"))
        filenameY = (os.path.join(self.folder_name_for_models,"Y.sav"))
        pickle.dump(Y, open(filenameY, "wb"))
        #procedure 2.
        #folder_name_for_backups = "backup"
        #Since X.sav and Y.sav will overide itself, tanking backup process as follows.
        self.create_folder_if_None_exists(self.folder_name_for_backups)
        d = self.get_currenttime_numeral()
        filenameX = (os.path.join(self.folder_name_for_backups,"X_{}.sav".format(d)))
        pickle.dump(X, open(filenameX, "wb"))
        filenameY = (os.path.join(self.folder_name_for_backups,"Y_{}.sav".format(d)))
        pickle.dump(Y, open(filenameY, "wb"))

    #Utilities below.
    def XYloader(self):
        """
        This one is not used amongst this sector or file. Use it by importing it from somewhere else.
        """
        filenameX = (os.path.join(self.folder_name_for_models,"X.sav"))
        with open(filenameX, mode="rb") as f:
            X = pickle.load(f)
        filenameY = (os.path.join(self.folder_name_for_models,"Y.sav"))
        with open(filenameY, mode="rb") as f:
            Y = pickle.load(f)
        return X,Y



#cwd = r"C:\Users\Andre\Pictureprocess\model"
#os.chdir(cwd)
if __name__ == "__main__":
    pass