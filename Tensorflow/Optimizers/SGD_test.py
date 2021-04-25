import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
###################################################################
###################################################################
x_train = [1, 2, 3]
y_train = [1, 2, 3]


train_data = np.array([[1, 2, 3]]) #input
train_label = np.array([[30, 30, 30]]) #label
X = tf.Variable(train_data, dtype=tf.float32, name='input')
Y_true = tf.Variable(train_label, dtype=tf.float32, name='input')
W1 = tf.Variable(tf.ones([3,4]), name='weight')
b1 = tf.Variable(tf.ones([1]), name='bias')
W2 = tf.Variable(tf.ones([4,3]), name='weight')
b2 = tf.Variable(tf.ones([1]), name='bias')



@tf.function
def cost():
    Y_1 = tf.matmul(X,W1) + b1
    Y_predict = tf.matmul(Y_1, W2) + b2
    print(Y_predict)
    error = tf.square(Y_true - Y_predict)
    return error


optimizer = tf.optimizers.SGD(learning_rate=0.01)
var_list = [[W1, b1], [W2, b2]]

#train = tf.keras.optimizers.Adam().minimize(cost, var_list)

grads_and_vars = optimizer._compute_gradients(cost, var_list=var_list)
optimizer.apply_gradients(grads_and_vars)

tf.print(W1)
tf.print(b1)
tf.print(W2)
tf.print(b2)



