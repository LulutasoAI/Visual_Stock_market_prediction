from PIL import Image
from Generator import Data_Generator
import os, glob
import sys 
from Labellingmachine import Labeller
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from TransferLearning import Transfer_learning
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Utils import Utils
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
import datetime 
from GeneralUtils import PictureProcessing
class Learn():
    def __init__(self,model_name="VGG16"):
        self.message = """
        run_mode 0 for newly generating X and Y while 
        run_mode 1 for loading the latest learning data.
        """
        self.save_folder_for_models = "models"
        self.backup_folder_for_models = "backup"
        Utils().create_folder_if_None_exists(self.save_folder_for_models)
        Utils().create_folder_if_None_exists(self.backup_folder_for_models)
        self.run_mode = int(sys.argv[1])
        
        self.model_name = model_name
        self.model_manager = Transfer_learning()
         
    def main(self):
        model = self.model_loader()
        if self.run_mode == 0:
            X,Y = self.data_praparation()
            PictureProcessing().XYpickler(X,Y)
        elif self.run_mode == 1:
            X,Y = PictureProcessing().XYloader()
        else:
            print(self.message)
            sys.exit()
        x_train, x_test, y_train, y_test = self.XnY2train(X,Y)
        print(x_train.shape)
        model = self.learning_process(x_train,y_train,model)
        Accuracy,Error_Rate,Precision,Recall,F_measure = self.general_validation_for_binary(model,x_test,y_test)
        self.model_save(model,self.save_folder_for_models,"latest_model.hdf5")
        self.model_save(model,self.backup_folder_for_models,"{}.hdf5".format(self.get_currenttime_numeral()))

    def data_praparation(self):
        X = []
        Y = []
        Labelling = Labeller()
        X_paths,Y_labels = Labelling.main()
        #Learning data preparation.
        for i, X_path in enumerate(X_paths):
            X.append(np.array(Image.open(X_path).convert('RGB').resize((256,256))).reshape(256,256,3))
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

    def XnY2train(self,X, Y, test_size =0.9, Shuffle = True):
        if shuffle == True:
            X,Y = shuffle(X, Y)
        else:
            pass
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=test_size)
        #map the files to a file
        xy = (x_train, x_test, y_train, y_test) #is this needed? probably not.
        return x_train, x_test, y_train, y_test

    def Train(self, x_train, y_train,model):
        cp = ModelCheckpoint("weights.hdf5", monitor="val_loss", verbose=1,
                     save_best_only=True, save_weights_only=True)
        optimizers = Adam(lr=0.00005, decay=1e-6)  #higher learning rate did not work well in my project. You can change it as you like.
        #SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
        results = {}
        epochs = 100
        model.compile(loss="sparse_categorical_crossentropy",  optimizer=optimizers, metrics=["accuracy"])
        results= model.fit(x_train, y_train,batch_size = 32, validation_split=0.3, epochs=epochs, shuffle=True,callbacks=[cp])
        return model

    def learning_process(self,x_train,y_train,model = None):
        if model == None:
            model = self.model_loader()
        model = self.Train(x_train,y_train,model)
        return model 
    
    def general_validation_for_multiple(self,model,x_test,y_test):
        predictions = model.predict(x_test)
        predictions_arged = []
        for i, prediction in enumerate(predictions):
            prediction = np.argmax(prediction)
            predictions_arged.append(prediction)
        #I think we put predictions_arged and y_test to the scikit-learn something.
        Precision = precision_score(y_test, predictions_arged, average='macro')
        Accuracy = accuracy_score(y_test, predictions_arged)
        Recall = recall_score(y_test, predictions_arged, average='macro')
        F_measure = f1_score(y_test, predictions_arged, average='macro')
        Error_Rate = 1-Accuracy
        for i, pred in enumerate(predictions_arged):
            #if i%15 == 0:
                #plt.imshow(x_test[i])
               # plt.title("Prediction : {}, Answer : {}".format(str(pred),y_test[i]))
               # plt.show()
            print("predicted : {}, ---------------vs-------------- answwer : {}".format(pred,y_test[i]))

        print("Accuracy : ",Accuracy)
        print("Error Rate : ", Error_Rate)
        print("Precision : ", Precision)
        print("Recall : ", Recall)
        print("F_measure : ", F_measure)

        return Accuracy,Error_Rate,Precision,Recall,F_measure

    def general_validation_for_binary(self,model, x_test, y_test):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        predictions = model.predict(x_test)
        predictions_arged = []
        #print(predictions,"predictions_whole")
        for i, prediction in enumerate(predictions):
            prediction = np.argmax(prediction)
            predictions_arged.append(prediction)
            if prediction == 0 and y_test[i] == 0:
                TP += 1
            elif prediction == 0 and y_test[i] == 1:
                FP += 1
            elif prediction == 1 and y_test[i] == 1:
                TN += 1
            elif prediction == 1 and y_test[i] ==0:
                FN += 1
            else:
                print("error occurred. at general_validation_TP part.")
                sys.exit()
        TPR = TP/(TP+FN)
        FPR = 1-(TN/(TN+FP))
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        Error_Rate = 1-Accuracy
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F_measure = 2/((1/Precision)+(1/Recall))
        print(predictions_arged)
        for i, pred in enumerate(predictions_arged):
            print("predicted : {}, ---------------vs-------------- answwer : {}".format(pred,y_test[i]))
        print("TRP : ",TPR)
        print("FPR : ", FPR)
        print("Accuracy : ",Accuracy)
        print("Error Rate : ", Error_Rate)
        print("Precision : ", Precision)
        print("Recall : ", Recall)
        print("F_measure : ", F_measure)
        #print("General Accuracy : ", (TP+TN)/len(y_test)) #It tunred out that General accuracy we calculated usually was the same as The official accuracy.


        return Accuracy,Error_Rate,Precision,Recall,F_measure

    
    def model_save(self, model, savefolder, modelname):
        model.save(os.path.join(savefolder,modelname))

    def model_load(self, base_model, savefolder, modelname):
        base_model.load_weights(os.path.join(savefolder,modelname))
        return base_model
    
    def get_currenttime_numeral(self):
        d = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return d
    

       


if __name__ == "__main__":
    learn = Learn()
    learn.main()
"""
inp = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((256,256))
        inp = np.array(inp).reshape(1,256,256,3)
        raw = self.model.predict(inp)
"""       
