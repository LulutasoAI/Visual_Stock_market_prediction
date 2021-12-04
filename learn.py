from PIL import Image
from Generator import Data_Generator
import os, glob
import sys 
from Labellingmachine import Labeller
import numpy as np
from matplotlib import pyplot as plt
class Learn():
    def __init__(self,model_name="VGG16"):
        pass 
    
    def data_praparation(self):
        X = []
        Y = []
        Labelling = Labeller()
        X_paths,Y_labels = Labelling.main()
        #以下勘でとりあえず書く 12/03
        for i, X_path in enumerate(X_paths):
            X.append(np.array(Image.open(X_path).convert('RGB').resize((256,256))).reshape(1,256,256,3))
            Y.append(1 if "down" in X_path else 0)
            #print(X_path,Y[i])
        X = np.array(X)
        Y = np.array(Y)
        
        print(Y[0])
        Image.fromarray(X[0].reshape(256,256,3)).show()
        plt.imshow(X[0].reshape(256,256,3))
        plt.show()
        print(Y)
        #print(len(X_paths),len(Y_labels))
        #ここまで 1203


if __name__ == "__main__":
    learn = Learn()
    learn.data_praparation()
"""
inp = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((256,256))
        inp = np.array(inp).reshape(1,256,256,3)
        raw = self.model.predict(inp)
"""       
