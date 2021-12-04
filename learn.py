from PIL import Image
from Generator import Data_Generator
import os, glob
import sys 
from Labellingmachine import Labeller

from TransferLearning import Transfer_learning
import numpy as np
from matplotlib import pyplot as plt
class Learn():
    def __init__(self,model_name="VGG16"):
        self.model_name = model_name
        self.model_manager = Transfer_learning()
         
    
    def data_praparation(self):
        X = []
        Y = []
        Labelling = Labeller()
        X_paths,Y_labels = Labelling.main()
        #Learning data preparation.
        for i, X_path in enumerate(X_paths):
            X.append(np.array(Image.open(X_path).convert('RGB').resize((256,256))).reshape(1,256,256,3))
            Y.append(1 if "down" in X_path else 0)
            #print(X_path,Y[i])
        X = np.array(X)
        Y = np.array(Y)
         """ #Mild test?
        print(Y[0])
        Image.fromarray(X[0].reshape(256,256,3)).show()
        plt.imshow(X[0].reshape(256,256,3))
        plt.show()
        print(Y)
        #print(len(X_paths),len(Y_labels))
        #ここまで 1203
        """
        return X, Y
    
    def model_loader(self):
        model = self.model_manager.load_model(self.model_name)
        return model 

    def XnY2train(self,X, Y, test_size =0.2, Shuffle = True):
        if shuffle == True:
            X,Y = shuffle(X, Y)
        else:
            pass
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=test_size)
        #map the files to a file
        xy = (x_train, x_test, y_train, y_test) #is this needed? probably not.
        return x_train, x_test, y_train, y_test

    def learning_process(self,X,Y):
        model = self.model_loader()
        #train_data prapared 
        x_train, x_test, y_train, y_test = XnY2train(X,Y)
        model = self.model_manager.Train(x_train,y_train,model)
        return model 
    
    def validation(self,x_test, y_test, model):
        predicts = []
        #The method below mihgt be insufficient but I am doing it for readability. 
        for i, x in enumerate(x_test):
            predicts.append(np.argmax(model.predict(x)))
        for i, predict in enumerate(predicts):
            true_ans = y_test[i]
            #some validation processes to calculate the F-value Precision, accurasy and so on.
            
            
        

    

       


if __name__ == "__main__":
    learn = Learn()
    learn.data_praparation()
"""
inp = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((256,256))
        inp = np.array(inp).reshape(1,256,256,3)
        raw = self.model.predict(inp)
"""       
