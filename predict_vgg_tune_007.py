'''
predict model
'''

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.models import Model

class predict_keras_vgg(object):

    def __init__(self,model_path):

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = Flatten(input_shape=base_model.output_shape[1:])(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(input=base_model.input, output=x)
        self.model.load_weights(model_path)
        print(self.model.summary())
        print(self.model.input)
        print(self.model.output)

    def predict(self,image_path):

        # --------------------------
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # print(x.shape)
        x = np.expand_dims(x, axis=0)
        y = self.model.predict(x)
        # print(y)
        return y

    def get_path_list(self,rootdir):
        '''
        :param rootdir: path 图片路径
        :return: file list
        '''
        FilePathList = []
        for fpathe, dirs, fs in os.walk(rootdir):
            for f in fs:
                FilePath = os.path.join(fpathe, f)
                if os.path.isfile(FilePath):
                    FilePathList.append(FilePath)
        return FilePathList



if __name__ == '__main__':

    a_predict_keras_vgg = predict_keras_vgg('./VGG_FeatureMap_Model.h5')
    FileList = a_predict_keras_vgg.get_path_list(r'C:\004_project\003_keras\keras_vgg_trans_learning\train_data\validation\cat')
    for file in FileList:
        print(file)
        print(a_predict_keras_vgg.predict(file))
