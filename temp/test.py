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