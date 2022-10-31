import sys

from keras.datasets import imdb

# load imdb data # the max word index is 9999
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)
# print(len(train_labels))
# print(train_labels[2])

# index dictionary; decoding the data and showing the comment
word_index = imdb.get_word_index()
# print(list(word_index.keys())[2:20])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_datas[2]])
# print(decoded_review)

# data preparation
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # build the zero matrix
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_datas)  # change the data to vector
x_test = vectorize_sequences(test_datas)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")  # change the type of the data

# build networks
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
# since the second layer, we don't need to declare the input_number
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers
from keras import losses
from keras import metrics
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# using optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# using custom losses and metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
#               loss=losses.binary_crossentropy,
#               metrics=['accuracy'])

# sampling to verify
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# show the results
import matplotlib.pyplot as plt

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf() # clean the image
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()