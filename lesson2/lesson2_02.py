import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

#data中有10个数据，每个数据为int，在[0,1000)以内
data = np.random.randint(1000, size = 10)

x = tf.constant(data, name = 'x')
y = tf.Variable(5 * x ** 2 - 3 * x + 15, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(x))
	print(session.run(y))
