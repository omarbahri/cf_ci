#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:12:08 2022

@author: omar
"""

import os
import numpy as np
import sys
from cvae import CVAE
import tensorflow as tf
import random
import time

tf.config.run_functions_eagerly(True)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
    
results_path = os.path.join(os.sep, root_dir, 'shapelet_aug', 'results', 
                            'ci', 'particles_spring')

name = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
nb_epochs = int(sys.argv[4])

data_path = os.path.join(root_dir, 'cf_ci', 'data', name) 

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
    
# nb_epochs = 1500

print("Loaded Dataset.."+str(name))

X_train = np.load(os.path.join(data_path, 'X_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
X_valid = np.load(os.path.join(data_path, 'X_valid.npy'))

# for dim in range(X_train.shape[1]):
#     mean = np.mean(X_train[:, dim, :])
#     std = np.std(X_train[:, dim, :])
#     X_train[:, dim, :] = (X_train[:, dim, :] - mean) / std
#     X_test[:, dim, :] = (X_test[:, dim, :] - mean) / std
#     X_valid[:, dim, :] = (X_valid[:, dim, :] - mean) / std

for dim in range(X_train.shape[1]):
    min_val = np.min(X_train[:,dim,:])
    max_val = np.max(X_train[:,dim,:])

# if min_val >= 0:
    X_train[:,dim,:] = (X_train[:,dim,:] - min_val) / (max_val - min_val)
    X_test[:,dim,:] = (X_test[:,dim,:]   - min_val) / (max_val - min_val)
    
# else:
#     for dim in range(X_train.shape[2]):
#         X_train[:,:,dim] = (X_train[:,:,dim] - min_val) / (max_val - min_val) * 2 - 1
#         X_test[:,:,dim] = (X_test[:,:,dim] - min_val) / (max_val - min_val) * 2 - 1
        

X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)
X_valid = tf.cast(X_valid, tf.float32)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
valid_size = X_valid.shape[0]

batch_size = 256

train_dataset = (tf.data.Dataset.from_tensor_slices(X_train)
                    .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(X_test)
                    .shuffle(train_size).batch(batch_size))
valid_dataset = (tf.data.Dataset.from_tensor_slices(X_valid)
                    .shuffle(train_size).batch(batch_size))

output_directory = os.path.join(results_path, name, 'cvae',
                'models_' + str(seed) + '_' + str(lr) + '_' + str(nb_epochs))

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], -1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[2], -1))

# if len(X_train.shape) == 2:  # if univariate
#     # add a dimension to make it multivariate with one dimension
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  # print('logpx_z', logpx_z)
  # print('logpz', logpz)
  # print('logqz_x', logqz_x)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
def reset_tf_graph():
    # Reset default graph
    tf.compat.v1.reset_default_graph()  # For TensorFlow 2.x, use tf.compat.v1

    # Clear any existing sessions
    tf.keras.backend.clear_session()
    if tf.executing_eagerly():
        # In TensorFlow 2.x with eager execution enabled
        tf.compat.v1.Session().close()
    else:
        # In TensorFlow 1.x or when eager execution is disabled
        tf.Session().close()

# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 16
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
   
print('lr', lr)

optimizer = tf.keras.optimizers.Adam(lr)
model = CVAE(latent_dim)

best_elbo = -np.inf

for epoch in range(1, nb_epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for valid_x in valid_dataset:
    loss(compute_loss(model, valid_x))
  elbo = -loss.result()
  
  if elbo > best_elbo:
    best_elbo = elbo
    # Save model weights
    model.save_weights(os.path.join(output_directory, 'best_model.hdf5'))

  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))

print('Final ELBO with', lr, ':', elbo)
print('Best ELBO with', lr, ':', best_elbo)


