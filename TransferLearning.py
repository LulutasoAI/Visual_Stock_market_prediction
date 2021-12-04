
import glob
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, Conv3D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from keras.layers.noise import AlphaDropout, GaussianDropout, GaussianNoise
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf


class Transfer_learning():
    def __init__(self):
        self.image_size = 256
        self.channel = 3
        self.base_model = VGG16(weights = 'imagenet',
            include_top = False,
            input_shape = (self.image_size, self.image_size, self.channel))

    def load_model(self,model_name):
        """
        It loads the model
        """
        if model_name == "VGG16":
            return self.load_Vgg16()
        elif model_name == "ResNet50v2":
            return self.load_ResNet50v2()

    def load_ResNet50v2(self):
        """
        load_ResNet50v2
        for tranining
        """
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape = (self.image_size, self.image_size, self.channel),
        pooling=None, classes=1000,
        classifier_activation='softmax'
        )
        base_model.summary()
        # 全ての重みを固定（freeze）
        untrainable_n = 30 #it could be 1 ~ 200 or so.
        for layer in base_model.layers[:untrainable_n]:
            layer.trainable = False

        # 重みが固定されているかの確認
        for layer in base_model.layers:
            print(layer, layer.trainable)

        model = keras.Sequential()
        # VGG16モデル
        model.add(base_model)
        # 層を追加
        model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        """
        #In case of loading some pretrained model.
        weights_path = "exetendedone.hdf5"
        model.load_weights(weights_path)
        """
        return model

    def load_Vgg16(self):
        """
        load VGG16
        for tranining
        """
        base_model = VGG16(weights = 'imagenet', #学習済みの重みを使用する
                           include_top = False, #出力層は使わない
                           input_shape = (self.image_size, self.image_size, self.channel)) #入力する画像サイズの指定
        base_model.summary()
        # 全ての重みを固定（freeze）
        for layer in base_model.layers[:19]:
          layer.trainable = False
        # 重みが固定されているかの確認
        for layer in base_model.layers:
            print(layer, layer.trainable)
        model = keras.Sequential()
        # VGG16モデル
        model.add(base_model)
        # 層を追加
        model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        """
        #In case of loading some pretrained model.
        weights_path = "exetendedone.hdf5"
        model.load_weights(weights_path)
        """
        return model

    def base_model_init_(self):
        base_model = self.base_model
        for layer in base_model.layers[:19]:
            layer.trainable = False
        #just for checking, delete if you want.
        for layer in base_model.layers:
            print(layer, layer.trainable)
        return base_model

    def model_merge(self, base_model):
        model = keras.Sequential()
        model.add(base_model)
        model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax')) #10 classes apparently
        model.summary()
        return model

    def Train(self, x_train, y_train,model):
        cp = ModelCheckpoint("weights.hdf5", monitor="val_loss", verbose=1,
                     save_best_only=True, save_weights_only=True)
        optimizers = Adam(lr=0.00005, decay=1e-6)  #higher learning rate did not work well in my project. You can change it as you like.
        #SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
        results = {}
        epochs = 100
        model.compile(loss="sparse_categorical_crossentropy",  optimizer=optimizers, metrics=["accuracy"])
        results= model.fit(x_train, y_train,batch_size = 20, validation_split=0.2, epochs=epochs, shuffle=True,callbacks=[cp])
        return model
