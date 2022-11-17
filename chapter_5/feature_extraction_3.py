# 5.3 little adjustment
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

train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')
test_gen = ImageDataGenerator(rescale=1./255)

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

# 上边的都没有动，下边开始微调
convnet.trainable=True
set_trainable = False
for l in convnet.layers:
    if l.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        l.trainable = True
    else:
        l.trainable = False

his = model.fit(train_generator,
                steps_per_epoch=100,
                epochs=100,
                batch_size=20,
                validation_data=validation_generator,
                validation_steps=50)

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['acc'])


from show_figure import show_figure, draw_smooth_curve

show_figure(his)
draw_smooth_curve(his)
