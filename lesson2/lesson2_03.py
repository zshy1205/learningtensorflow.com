import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x =  tf.Variable(0, name = 'x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	for i in range(5):
		x = x + 1
		print(session.run(x))