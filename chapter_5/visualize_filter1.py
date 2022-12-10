from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def deprocess_image(x):
    """对张量做标准化，使其均值为0，标准差为0.1"""
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    """裁切到[0, 1]"""
    x += 0.5
    x = np.clip(x, 0, 1)
    """裁切到[0, 255]"""
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


# import model
model = VGG16(weights='imagenet', include_top=False)
model.summary()

max_size = 2
layer_name = 'block1_conv1'
size = 64
margin = 2

results = np.zeros((max_size*size+(max_size-1)*margin,
                    max_size*size+(max_size-1)*margin, 3))

for i in range(max_size):
    for j in range(max_size):
        filter_img = generate_pattern(layer_name, i+(j*max_size), size=size)
        # plt.imshow(filter_img)
        # plt.show()
        print(type(filter_img[0, 0, 0]))
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        # print("{}, {}".format(horizontal_start, horizontal_end))
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
print(type(results[0, 0, 0]))
plt.figure(figsize=(10, 10))
plt.imshow(results/255)
plt.show()



