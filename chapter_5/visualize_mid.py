from keras.models import load_model

model = load_model("cats_vs_dogs_small_1.h5")
model.summary()

from keras.utils import image_utils
import numpy as np

img_path = "/home/zlh/data/dogs_vs_cats/test1/3229.jpg"
img = image_utils.load_img(img_path, target_size=(150, 150))
img_tensor = image_utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
xxx = models.Model()