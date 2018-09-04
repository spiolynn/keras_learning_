'''
将图片数据，生成为npy数据 yield方式
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

datagen = ImageDataGenerator()

model = VGG16(weights='imagenet', include_top=False)

i = 1
for batch,label in datagen.flow_from_directory('data/train',
                                               target_size=(224, 224),
                                               batch_size=1,
                                               class_mode='binary',
                                               shuffle=False,
                                               classes=['dog','cat']):
    print(type(batch),batch.shape)
    print(type(label), label.shape)
    print('label' + str(label[0]))
    ## 生成npy文件

    # x = np.expand_dims(batch, axis=0)
    x = preprocess_input(batch)
    features = model.predict(x)
    print(features.shape)

    filename = r'data_np//train//' + str(i) + '_' +str(int(label[0]))+'.npz'
    print(filename)
    np.savez(filename,features=features,label=label)
    i += 1
    if i > 2:
        break