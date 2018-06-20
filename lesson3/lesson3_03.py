import matplotlib.image as mping
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/photo.jpg'

image = mping.imread(filename)

x = tf.Variable(image, name = 'x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
	shape = tf.shape(x)
	sess.run(model)
	shape = sess.run(shape)
	#左旋90度
	#x = tf.transpose(x, perm=[1,0,2])
	#左右翻转
	#x = tf.reverse_sequence(x, [int(height / 2)] * width, 1, batch_dim = 0)
	#x = tf.reverse_sequence(x, [height] * width, 1, batch_dim = 0)
	x = tf.reverse_sequence(x, np.ones((shape[1],)) * shape[0], 0, batch_dim = 1)
	result = sess.run(x)

plt.imshow(result)
plt.show()