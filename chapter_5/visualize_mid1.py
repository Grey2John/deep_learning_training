# P134 show every channel of the picture 两个维度显示 所有激活输出 所有的通道
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
# print(img_tensor.shape)

import matplotlib.pyplot as plt

# plt.imshow(img_tensor[0])

from keras.models import Model

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
# print(first_layer_activation)

# name the layers
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(((size + 1) * n_cols - 1,
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1): (row + 1) * size + row] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")

# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = int(layer_activation.shape[-1])
#
#     size = int(layer_activation.shape[1])
#     n_cols = n_features
#     print("n_cols: {}".format(n_cols / images_per_row))
#     display_grid = np.zeros((int(n_cols / images_per_row) * size, images_per_row * size))
#     # 矩阵合成
#
#     for col in range(int(n_cols / images_per_row)):
#         for row in range(images_per_row):
#             channel_index = int(col * images_per_row) + row
#             print(channel_index)
#             # print("{}*{}+{}={}".format(col, images_per_row, row, col * images_per_row + row))
#             channel_image = layer_activation[0, :, :, channel_index].copy
#             if channel_image.sum() != 0:
#                 channel_image -= channel_image.mean()
#                 channel_image /= channel_image.std()
#                 channel_image *= 64
#                 channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size: (col + 1) * size + col,
#                          row * size: (row + 1) * size + row] = channel_image
#
#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect="auto", cmap="viridis")
plt.show()
