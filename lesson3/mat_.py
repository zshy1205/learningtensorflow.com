import matplotlib.image as mping
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/photo.jpg'

#load the image
image = mping.imread(filename)
print(image.shape)
height, width, depth = image.shape
#plt.imshow(image)
#plt.show()

#转置图片
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
	#tf.transpose(x, perm=[1,0,2])的作用是将输入张量的第一维度与第二维度交换
	#所以在这里可以逆时针旋转90°
	#x = tf.transpose(x, perm=[1,0,2])
	#像素翻转
	x = tf.reverse_sequence(x, [width] * height, 1, batch_dim = 0)
	sess.run(model)
	result = sess.run(x)

plt.imshow(result)
plt.show()