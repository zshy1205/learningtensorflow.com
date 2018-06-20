import matplotlib.image as mping
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/photo.jpg'

image = mping.imread(filename)
height, width, depth = image.shape
x = tf.Variable(image, name='x')
mirrored = tf.Variable(image, name='mirrored')
model = tf.initialize_all_variables()

mirror_mask = np.ones((height,)) * (width/2)


with tf.Session() as session:
    # Note swapped dims in the last two parameters
    #从每一行的第0列到中间列进行左右翻转
    mirrored = tf.reverse_sequence(x, mirror_mask, 1, batch_dim=0)

    # Now stich them back up again


    session.run(model)
    result = session.run(mirrored)

print(result.shape)
plt.imshow(result)
plt.show()