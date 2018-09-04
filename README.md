# keras_learning_
keras_learning_

## 010-Keras-VGG预训练-数据增强-迁移学习

### 0 引用

> https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html <br> Keras 官方面向小数据集使用keras基于VGG-16微调的图像分类

> https://www.ctolib.com/topics-126305.html <br> ctolib小哥中文版

### 1 Why

> CNN模型实际是完成对图像数据特征提取的工作(取代ML特征工程)，不同的CNN模型的优劣，实际是对图像特征表征的优劣。

> CNN模型 特征表征 好坏 取决于 模型设计本身和训练数据

> 菜鸟折腾出来的CNN模型weights，常常会比大牛们预训练好的模型表现要差

> 当我只有少量训练数据时，又是菜鸟，不如使用域训练模型来帮助我进行特征提取，提高训练效率。

### 2 数据(使用1000张猫狗照片区分)

> https://www.kaggle.com/c/dogs-vs-cats/data <br> 获取训练数据

> https://pan.baidu.com/s/1pLANsoF#list/path=%2F <br> 百度云盘数据

![image.png](https://upload-images.jianshu.io/upload_images/10357485-acbba3001fe0a49b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/10357485-b9e28a80fea7aac8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 3 第一次模型设计（LeCun的LEnet）

![graph_large_attrs_key=_too_large_attrs&limit_attr_size=1024&run=.png](https://upload-images.jianshu.io/upload_images/10357485-34cf900a36936115.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001/jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001/jpg
            cat002.jpg
            ...
```

```
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# nb_train_samples // batch_size 整除
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[TensorBoard(log_dir='./log_dir')])

model.save_weights('first_try.h5')
```

> 1 小数据集上-常容易过拟合-使用数据提升来对抗之

![image](http://keras-cn.readthedocs.io/en/latest/images/cat_data_augmentation.png)

> 2 关注的是模型的“熵容量”——模型允许存储的信息量。**能够存储更多信息的模型能够利用更多的特征取得更好的性能，但也有存储不相关特征的风险。另一方面，只能存储少量信息的模型会将存储的特征主要集中在真正相关的特征上，并有更好的泛化性能。**

- 熵容量调整:
    - 1 调整模型的参数数目（模型层数、每层的规模）
    - 2 权重进行正则化约束 (L1 L2)
    - 3 Dropout


```
120/125 [===========================>..] - ETA: 5s - loss: 0.6692 - acc: 0.5938
121/125 [============================>.] - ETA: 4s - loss: 0.6683 - acc: 0.5945
122/125 [============================>.] - ETA: 3s - loss: 0.6667 - acc: 0.5963
123/125 [============================>.] - ETA: 2s - loss: 0.6670 - acc: 0.5971
124/125 [============================>.] - ETA: 1s - loss: 0.6703 - acc: 0.5973
125/125 [==============================] - 285s 2s/step - loss: 0.6699 - acc: 0.5985 - val_loss: 0.6032 - val_acc: 0.6786
Epoch 3/50

119/125 [===========================>..] - ETA: 6s - loss: 0.4237 - acc: 0.8188
120/125 [===========================>..] - ETA: 5s - loss: 0.4227 - acc: 0.8187
121/125 [============================>.] - ETA: 4s - loss: 0.4247 - acc: 0.8171
122/125 [============================>.] - ETA: 3s - loss: 0.4273 - acc: 0.8171
123/125 [============================>.] - ETA: 2s - loss: 0.4271 - acc: 0.8176
124/125 [============================>.] - ETA: 1s - loss: 0.4258 - acc: 0.8180
125/125 [==============================] - 258s 2s/step - loss: 0.4268 - acc: 0.8175 - val_loss: 0.8721 - val_acc: 0.7047
```

![image.png](https://upload-images.jianshu.io/upload_images/10357485-b0133d628d9cb317.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/10357485-5bd78ae426285366.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 3.1 第一次模型结果分析

- 1 用时3.5h 
- 2 val_acc acc 具有明显差异，过拟合
- 3 特征表述不算好，acc得分在0.8几分


### 4 使用VGG模型来进行特征工程

> 这部分实现见 `101-Keras-如何做到边导数边训练`

### 5 在预训练的网络上fine-tune


```python
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers as optimizers

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# top_model = Sequential()
# top_model.add(Flatten(input_shape=(7,7,512)))
# top_model.add(Dense(256))
# top_model.add(Activation('relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1))
# top_model.add(Activation('sigmoid'))


base_model = VGG16(weights='imagenet', include_top=False,input_shape = (224, 224, 3))
x = base_model.output
print(base_model.output_shape)

# inputs = Input(shape = model.output_shape[1:])
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(input=base_model.input, output=x)

# print(model.output_shape[1:])
print(model.summary())

for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# nb_train_samples // batch_size 整除
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[TensorBoard(log_dir='./log_dir')])

model.save_weights('first_try.h5')
```

- 10 轮训练完成后结果

```
240/250 [===========================>..] - ETA: 5:34 - loss: 0.0324 - acc: 0.9880
241/250 [===========================>..] - ETA: 5:01 - loss: 0.0324 - acc: 0.9881
242/250 [============================>.] - ETA: 4:27 - loss: 0.0323 - acc: 0.9881
243/250 [============================>.] - ETA: 3:54 - loss: 0.0323 - acc: 0.9882
244/250 [============================>.] - ETA: 3:20 - loss: 0.0326 - acc: 0.9877
245/250 [============================>.] - ETA: 2:47 - loss: 0.0325 - acc: 0.9878
246/250 [============================>.] - ETA: 2:14 - loss: 0.0326 - acc: 0.9878
247/250 [============================>.] - ETA: 1:40 - loss: 0.0325 - acc: 0.9879
248/250 [============================>.] - ETA: 1:07 - loss: 0.0324 - acc: 0.9879
249/250 [============================>.] - ETA: 33s - loss: 0.0322 - acc: 0.9880
250/250 [==============================] - 9984s 40s/step - loss: 0.0322 - acc: 0.9880 - val_loss: 0.1534 - val_acc: 0.9575
Epoch 11/20
```
