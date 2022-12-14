from keras.applications import VGG16
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet', include_top=False)
model.summary()
layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]  ###

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

"""初始输入图像为一张带有噪声的灰度图像"""
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

"""每次梯度更新的步长"""
step = 1

"""运行40次梯度上升"""
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step # 沿着损失最大化的方向更新


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


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(generate_pattern('block3_conv1', 0))

plt.show()