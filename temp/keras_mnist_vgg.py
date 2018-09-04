from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
from PIL import Image
from keras.callbacks import TensorBoard

import sys
import os
sys.path.append(os.getcwd())
print(os.getcwd())

from npy_training import generate_batch_data_random

# step1- 数据
data_path = 'data_np/train'
valid_path = 'data_np/validation'
batch_size = 8

# step2- 模型设计
model = VGG16(weights='imagenet', include_top=False)
model = Sequential()
model.add(Flatten(input_shape=(7,7,512)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
        generate_batch_data_random(data_path,batch_size),
        samples_per_epoch=2000//batch_size,
        nb_epoch=20,
        validation_data=generate_batch_data_random(valid_path,batch_size),
        nb_val_samples=400//batch_size,
        callbacks = [TensorBoard(log_dir='./log_dir')]
        )

model.save_weights('VGG_MNIST.h5')



