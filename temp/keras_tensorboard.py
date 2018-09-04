'''
keras模型可视化
'''

# keras learning
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard

###################### data
data = np.ones((100,784))
labels = np.zeros((100,10))
######################

################################## model
# 这部分返回一个张量
inputs = Input(shape=(784,))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels,callbacks=[TensorBoard(log_dir='./log_dir')])  # 开始训练
################################## model

################################## tensorboard