'''
生成feature map
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import os

class vgg_feature_map(object):

    def __init__(self):

        self.datagen = ImageDataGenerator()
        self.model = VGG16(weights='imagenet', include_top=False)

        print('model summary\n')
        print(self.model.summary())
        print(self.model.input)
        print(self.model.output)

    def generate_feature_map(self,train_data_path,target_size,target_path,data_num):

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        i = 1
        for batch, label in self.datagen.flow_from_directory(train_data_path,
                                                        target_size=target_size,
                                                        batch_size=1,
                                                        class_mode='binary',
                                                        shuffle=False
                                                        #classes=['dog', 'cat']
                                                             ):
            # print(type(batch), batch.shape)
            # print(type(label), label.shape)
            # print('label' + str(label[0]))
            ## 生成npy文件

            # x = np.expand_dims(batch, axis=0)
            x = preprocess_input(batch)
            features = self.model.predict(x)
            print(features.shape)

            filename = str(i) + '_' + str(int(label[0])) + '.npz'
            targetpath = os.path.join(target_path,filename)
            print(targetpath)

            np.savez(targetpath, features=features, label=label)
            i += 1
            if i > data_num:
                break

if __name__ == '__main__':
    pass
    a_vgg_feature_map = vgg_feature_map()
    # a_vgg_feature_map.generate_feature_map('train_data\\train',(224,224),'train_np\\train',2000)
    # a_vgg_feature_map.generate_feature_map('train_data\\validation', (224, 224), 'train_np\\validation', 400)



