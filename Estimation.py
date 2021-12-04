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


class estimation():
    def __init__(self):
        self.the_latest_model_name = "latest_model.hdf5"

