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
from keras.callbacks import ModelCheckpoint

'''
monitor=’val_acc’:这是我们所关心的度量：验证精确度
verbose=1:它将打印更多信息
save_best_only=True:只保留最好的检查点(在最大化验证精确度的情况下)
mode=’max’:以最大化验证精确度保存检查点
'''
filepath="weights-tune-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
 monitor='val_acc',
 verbose=1,
 save_best_only=True,
 mode='max',
 period=1)

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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
    callbacks=[TensorBoard(log_dir='./tensorboard'),checkpoint])

model.save('vgg_fine_tune.h5')