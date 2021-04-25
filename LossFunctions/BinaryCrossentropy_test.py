# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

y_true = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
y_pred = np.array([0.2, 0.8, 0.7, 0.9, 0.8, 0.8, 0.6, 0.8, 0.9, 0.8])

bce = tf.keras.losses.BinaryCrossentropy()
bce_loss = bce(y_true, y_pred).numpy()
print('bce_loss:', bce_loss)

my_loss = - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
mean_my_loss = np.mean(my_loss)
print('my_loss:', mean_my_loss)




