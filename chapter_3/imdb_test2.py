from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # build the zero matrix
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_datas)  # change the data to vector
x_test = vectorize_sequences(test_datas)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")  # change the type of the data

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=512)

results = model.evaluate(x_test, y_test)
print(model.predict(x_test))