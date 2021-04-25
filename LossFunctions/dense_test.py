# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

x_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
model = tf.keras.Sequential()
model.add(layers.Dense(kernel_initializer='OnesV2', bias_initializer='OnesV2', units=3, input_shape=(1, 4)))
y_pre = model(x_tf)
print(y_pre)
model.add(layers.Dense(kernel_initializer='OnesV2', bias_initializer='OnesV2', units=2))
y_pre = model(x_tf)
print(y_pre)









