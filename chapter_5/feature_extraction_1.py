# 5.3 fast feature extraction without data enhancement
import sys

from keras.applications import VGG16

convnet = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))
convnet.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/zlh/data/dogs_vs_cats/part_data_path'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'verify')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(dire, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    all_labels = np.zeros(shape=sample_count)
    generator = datagen.flow_from_directory(
        dire,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    i = 0
    for inputs, labels_batch in generator:
        features_batch = convnet.predict(inputs)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        all_labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break
    return features, all_labels

train_features, train_labels = extract_features(train_dir, 2000)
val_features, val_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4*4*512))
val_features = np.reshape(val_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00002),
              loss='binary_crossentropy',
              metrics=['acc'])
his = model.fit(train_features, train_labels,
                epochs=30,
                batch_size=20,
                validation_data=(val_features, val_labels))


# show results
import matplotlib.pyplot as plt

acc = his.history['acc']
val_acc = his.history['val_acc']
loss = his.history['loss']
val_loss = his.history['val_loss']

epochs = range(1, len(acc)+1)
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