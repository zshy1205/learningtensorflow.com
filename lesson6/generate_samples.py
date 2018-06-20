import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters, seed = seed)

model = tf.global_variables_initializer()
with tf.Session() as session:
    for i in range(1):
    	nearest_indices = assign_to_nearest(samples, initial_centroids)
    	centroids = update_centroids(samples, nearest_indices, n_clusters) 
    sample_values = session.run(samples)
    updated_centroid_value = session.run(centroids)
    
print(updated_centroid_value)
print(initial_centroids)
plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)