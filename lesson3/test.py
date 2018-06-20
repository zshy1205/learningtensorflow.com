import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable([[1,2,3], [4,5,6]], name = 'x')

t1 = tf.constant([1,2,3])
t2 = tf.constant([3,4,5])
concated = tf.concat([t1, t2], 0)

print(np.ones(3,))
print(np.ones(3,) * 4)
print(np.ones((3,)) * 4)
model = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(model)
	print(sess.run(x))
	x = tf.transpose(x, perm=[1,0])
	print(sess.run(x))
	print(sess.run(concated))