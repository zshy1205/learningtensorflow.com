import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 100, name='y')

with tf.Session() as session:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("tmp/basic", session.graph)
	model = tf.global_variables_initializer()
	session.run(model)
	print(session.run(y))