#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:42:21 2024

@author: omar
"""

import tensorflow as tf

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(20, 49, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
    
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=5*13*32, activation=tf.nn.relu),  # Adjusted units for new input shape
                tf.keras.layers.Reshape(target_shape=(5, 13, 32)),  # Adjusted target shape
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=2, strides=2, padding='valid', activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=2, strides=2, padding='valid', activation='relu'),
                # Final Conv2DTranspose layer to match the input shape
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 4), strides=(1, 1), padding='valid', activation=None)
            ]
        )

    @tf.function
    def sample(self, eps=None):
      if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
      return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
      mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
      return mean, logvar
    
    def reparameterize(self, mean, logvar):
      eps = tf.random.normal(shape=mean.shape)
      return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
      logits = self.decoder(z)
      if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
      return logits
  
    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)
    
