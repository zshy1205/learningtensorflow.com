import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mean = tf.Variable(0., name='mean')
n = tf.Variable(0., name='n')

model = tf.global_variables_initializer()

m = 10
max_ = 100

with tf.Session() as session:
	for i in range(2):
		new_random_number = np.random.randint(max_, size = m)
		sum_of_random_number = np.sum(new_random_number)
		print(sum_of_random_number)

		n = n + m
		mean = (mean * (n - m) + sum_of_random_number) / n

	session.run(model)
	print(session.run(mean))
#求多个array的mean值

