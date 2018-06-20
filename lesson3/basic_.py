import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(3, name='a')
b = tf.constant(4, name='b')
add_op = a + b

a1 = tf.constant([1,2,3], name='a1')
b1 = tf.constant([4,5,6], name='b1')
add_op_1 = a1 + b1
add_op_2 = a1 + b

a2 = tf.constant([[1,2,3], [4,5,6]], name='a2')
b2 = tf.constant([[1,2,3], [4,5,6]], name='b2')
add_op_3 = a2 + b2
add_op_4 = a2 + b
add_op_5 = a2 + b1
c = tf.constant([[100],[1000]], name='c')
add_op_6 = a2 + c

model = tf.global_variables_initializer()

with tf.Session() as sess:
	print(sess.run(add_op))
	print(sess.run(add_op_1))
	print(sess.run(add_op_2))
	print(sess.run(add_op_3))
	print(sess.run(add_op_4))
	print(sess.run(add_op_5))
	print(sess.run(add_op_6))