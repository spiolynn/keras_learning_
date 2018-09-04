# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Feb  5 15:31:25 2018

@author: brucelau
"""

'''Trains a simple deep NN on the MNIST dataset.'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
import numpy as np

'''
导入
'''
def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


batch_size = 552
num_classes = 10
epochs = 2
DROPOUT = 0.5
opt='adam'
patience = 4

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



## 模型定义
model = Sequential()
with tf.name_scope('layer-1'):
    model.add(Dense(500, activation='relu', input_shape=(784,),name='H1'))
    model.add(Dropout(DROPOUT,name='DO1'))
with tf.name_scope('layer-2'):
    model.add(Dense(300, activation='relu',name='H2'))
    model.add(Dropout(DROPOUT,name='DO2'))
with tf.name_scope('output'):
    model.add(Dense(num_classes, activation='softmax',name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#%%
# callback on_train_run as a Class
class Mylogger(keras.callbacks.Callback):
    def on_train_begin(self,logs=None):
        print('001-On_train_begin')
        # model.summary()
        print('model summary -------------------------')
        print(keras.utils.layer_utils.print_summary(self.model))

# callback loss-show
show_loss_callback = LambdaCallback(on_epoch_end = lambda epoch,logs:
    print(epoch,logs['loss'],type(epoch),type(logs['loss'])))

# callback loss-plot
def vis(e,l):
    plt.figure(1)
    plt.scatter(e,l)
    plt.xlabel('epochs')
    plt.ylabel('train-accuracy')
    plt.legend()
    plt.title('The training process')
    plt.show()

plot_loss_callback = LambdaCallback(on_epoch_end = lambda epoch,logs:
    vis(epoch,logs['loss']))

# recording loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
    def vis_losss(self):
        plt.figure(2)
        plt.plot(np.arange(len(self.losses)),self.losses,label='losses')
        plt.plot(np.arange(len(self.val_losses)),self.val_losses,label='val_losses')
        plt.xlabel('epochs')
        plt.ylabel('train-accuracy')
        plt.legend()
        plt.title('The training process')
#%%
history = LossHistory()

# 模型可视化
tbCallBack = keras.callbacks.TensorBoard(log_dir='./log_dir',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

#%%
model_history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test),
                    callbacks = [Mylogger(),
                                 tbCallBack,
                                 ])

# model_history = model.fit(x_train,
#                     y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=0,
#                     validation_data=(x_test, y_test),
#                     callbacks = [Mylogger(),
#                                  tbCallBack,
#                                  EarlyStopping(patience=patience,mode='min',verbose=0),
#                                  show_loss_callback,
#                                  plot_loss_callback,
#                                  history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# history.vis_losss()

'''
C:\ProgramData\Anaconda3\envs\lemon\Scripts\tensorboard.exe --logdir C:\Users\hupan-wk\PycharmProjects\keras_project\keras_001_draw\log_dir
'''