import numpy as np
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

def get_vgg():
    bottom_model = VGG16(weights='imagenet', include_top=False, input_shape=(w, h, 3))

    n_cnn_out = bottom_model.output_shape[1:]
    n_fcc_in = n_cnn_out[0] * n_cnn_out[1] * n_cnn_out[2]

    # construct FC layer
    top_model = Sequential()

    top_model.add(Flatten(input_shape=n_cnn_out))
    top_model.add((Dense(32, activation='relu')))
    top_model.add((Dense(32, activation='relu')))
    top_model.add(Dense(6, activation="softmax"))

    # Connect CNN and FCC layer
    model = Model(input=bottom_model.input, output=top_model(bottom_model.output))

    # 最後のconv層の直前までの層をfreeze
#     for layer in model.layers[:15]:
#         layer.trainable = False

    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=["accuracy"])
    return model
    
model = get_vgg()


x_train_rgb = np.tile(x_train, 3)

# tb_cb = callbacks.TensorBoard(log_dir='train/', histogram_freq=1,  write_graph=False)
model.fit(x_train_rgb, y_train_hot, batch_size=64, shuffle=True, epochs=20)

model.save('models/vgg_finetune.h5')
