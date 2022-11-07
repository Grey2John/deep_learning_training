from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=['acc'])

# data processing
from keras.preprocessing.image import ImageDataGenerator as IDG
train_datagen = IDG(rescale=1./255)  # declare the range of element
test_datagen = IDG(rescale=1./255)

train_dir = '/home/zlh/data/dogs_vs_cats/'
train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=(150, 150),
                                              batch_size=20,
                                              class_mode='binary')
