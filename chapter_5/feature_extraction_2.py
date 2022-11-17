# 5.3 feature extraction with data enhancement
import sys

from keras.applications import VGG16

convnet = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))
convnet.summary()

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(convnet)
model.add(layers.Flatten())  # 多维输入一维化
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
convnet.trainable = False

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00002),
              loss='binary_crossentropy',
              metrics=['acc'])

# data enhancement
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/zlh/data/dogs_vs_cats/part_data_path'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'verify')
test_dir = os.path.join(base_dir, 'test')

train_gen = ImageDataGenerator(rescale=1. / 255,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')
test_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_gen.flow_from_directory(train_dir,
                                                target_size=(150, 150),
                                                batch_size=20,
                                                class_mode='binary')

validation_generator = test_gen.flow_from_directory(validation_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

his = model.fit(train_generator,
                steps_per_epoch=100,
                epochs=30,
                batch_size=20,
                validation_data=validation_generator,
                validation_steps=50)

# show results
import matplotlib.pyplot as plt

acc = his.history['acc']
val_acc = his.history['val_acc']
loss = his.history['loss']
val_loss = his.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'ro', label='traning loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
