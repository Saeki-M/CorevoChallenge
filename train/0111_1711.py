import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import SGD, Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.utils import np_utils
from keras.layers.core import Lambda

from keras.utils.vis_utils import model_to_dot
from keras import callbacks

import tensorflow as tf

from keras.models import load_model
import pickle

from sklearn import preprocessing
from keras.utils import np_utils
from tqdm import tnrange, tqdm_notebook


# load data
x_train = np.expand_dims(np.load('corevo/features/spectrogram/x_train.npy'), 3)
y_train = np.load('corevo/features/spectrogram/y_train.npy')


with open('corevo/features/spectrogram/validation.pkl', mode='rb') as f:
    validation = pickle.load(f)
    
N, w, h, _ = x_train.shape
y_train_hot = np_utils.to_categorical(y_train)

label_dict = {'MA_CH':0, 'MA_AD':1, 'MA_EL':2,'FE_CH':3, 'FE_AD':4, 'FE_EL':5}


# load model

def get_model():
    model = Sequential()
    model.add(Conv2D(20, (4, 4), strides=(1,1),  input_shape=(w, h, 1), activation='relu', name='first_conv_layer'))
    model.add(Conv2D(20, (4, 4), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(10, (4, 4), strides=(1, 1), activation='relu'))
    model.add(Conv2D(5, (3, 3), strides=(1, 1), activation='relu', name='last_conv_layer'))
    model.add(MaxPool2D())
    
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = get_model()
model.summary()


model.fit(x_train, y_train_hot, batch_size=100, shuffle=True, epochs=20)
model.save('~/corevo-challenge/models/0111_1711.h5')
