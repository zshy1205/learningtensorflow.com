import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([23,34,56], name='x')
y = tf.Variable(x + 5, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))
	y = y + 15
	y = y + 100
	print(session.run(y))