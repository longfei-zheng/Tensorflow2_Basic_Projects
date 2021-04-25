import tensorflow as tf

opt = tf.keras.optimizers.Adam(learning_rate=0.2)
var1 = tf.Variable(10.0)
loss = lambda: (var1 ** 2) / 2.0  # d(loss)/d(var1) == var1
step_count = opt.minimize(loss, [var1]).numpy()
# The first step is `-learning_rate*sign(grad)`
print(var1.numpy())

grads_and_vars = opt._compute_gradients(loss, var_list=var1)
opt.apply_gradients(grads_and_vars)
print(var1.numpy())

###################################################################
###################################################################
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random.normal([1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')
hypothesis = W*x_train+b

#@tf.function
def cost():

    y_model = W*x_train+b
    error = tf.reduce_mean(tf.square(y_train- y_model))
    return error


optimizer = tf.optimizers.SGD(learning_rate=0.01)
var_list=[W, b]
train = tf.keras.optimizers.Adam().minimize(cost, var_list)

grads_and_vars = optimizer._compute_gradients(cost, var_list=var_list)
optimizer.apply_gradients(grads_and_vars)

tf.print(W)
tf.print(b)





