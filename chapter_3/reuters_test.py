import sys

from keras.datasets import reuters
import numpy as np
# load data
(train_datas, train_labels), (test_datas, test_labels) = reuters.load_data(num_words=10000)

# data processing & encoding data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_datas)
x_test = vectorize_sequences(test_datas)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# build the networks
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# verify dataset
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# results visualization
import matplotlib.pyplot as plt

loss_val = history.history['val_loss']
loss = history.history['loss']
print("the length of loss is: {}".format(len(loss_val)))
epochs = range(1, len(loss_val) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, loss_val, 'r', label='validation loss')
plt.title('Trianing and validation loss')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label="training acc")
plt.plot(epochs, val_acc, 'r', label="validation acc")
plt.title("training and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()