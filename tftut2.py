#tensorflow tutorial 2
#Faiz ul haque Zeya
import numpy as np
import tensorflow as tf

# Weight and bias for neural networks
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# Adagrad optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4,5,6,7,8,9,10]
y_train = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("Adagrad Optimizers W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4,5,6,7,8,9,10]
y_train = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W1, curr_b1, curr_loss1 = sess.run([W, b, loss], {x:x_train, y:y_train})
print("Adadelta Optimizers W: %s b: %s loss: %s"%(curr_W1, curr_b1, curr_loss1))
print("The difference in loss of Adagrad and Adadelta is %s"% (curr_loss - curr_loss1))
