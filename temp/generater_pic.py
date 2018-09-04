'''
生成图片
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen = ImageDataGenerator()

img = load_img('datu-1.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
y = np.array(['1'],dtype='int')


#============================================================
'''
y = None batch is (1, 340, 260, 3) numpy
y != None batch[0] is (1, 340, 260, 3) numpy batch[1] (1,)
'''
for batch in datagen.flow(x,y,batch_size=1,
                          save_to_dir='test', save_prefix='cat', save_format='jpeg'):
    print(type(batch))
    print(batch[0].shape)
    print(batch[1].shape)
    i += 1
    if i > 1:
        break  # otherwise the generator would loop indefinitely
#============================================================

#============================================================
# for batch,label in datagen.flow_from_directory('test',target_size=(150, 150),batch_size=1):
#     print(type(batch))
#     print(type(label))
#     i += 1
#     if i > 1:
#         break
#============================================================