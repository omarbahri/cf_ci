#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:35:28 2024

@author: omar
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def _energy(loc, vel, edges, interaction_strength):
    num_simulations, length_of_timeseries, _, num_balls = vel.shape
    
    energy_train = np.zeros((num_simulations, length_of_timeseries, num_balls))
    
    # Compute the kinetic and potential energy for each particle at each time step
    for sim in range(num_simulations):
        for t in range(length_of_timeseries):
            K = 0.5 * (vel[sim, t] ** 2).sum(axis=0)  # Kinetic energy of each ball
            
            U = np.zeros(num_balls)  # Potential energy contribution for each ball
            for i in range(num_balls):
                for j in range(num_balls):
                    if i != j:
                        r = loc[sim, t, :, i] - loc[sim, t, :, j]
                        dist = np.sqrt((r ** 2).sum())
                        U[i] += (
                            0.5
                            * interaction_strength
                            * edges[sim, i, j]
                            * (dist ** 2)
                            / 2
                        )
            # Sum kinetic and potential energy for each ball
            energy_train[sim, t] = K + U
    
    return energy_train

interaction_strength = 0.1

n_balls = 5

train_size = ''
valid_size = ''
test_size = ''

if train_size == '':
    _s = ''
else:
    _s = '_s'

loc_train = np.load(os.path.join('data', 'loc_train_springs' + str(n_balls) +\
                                 _s + str(train_size) + '.npy'))
edges_train = np.load(os.path.join('data', 'edges_train_springs' + str(n_balls) +\
                                 _s + str(train_size) + '.npy'))
vel_train = np.load(os.path.join('data', 'vel_train_springs' + str(n_balls) +\
                                 _s + str(train_size) + '.npy'))
    
loc_valid = np.load(os.path.join('data', 'loc_valid_springs' + str(n_balls) +\
                                 _s + str(valid_size) + '.npy'))
edges_valid = np.load(os.path.join('data', 'edges_valid_springs' + str(n_balls) +\
                                 _s + str(valid_size) + '.npy'))
vel_valid = np.load(os.path.join('data', 'vel_valid_springs' + str(n_balls) +\
                                 _s + str(valid_size) + '.npy'))
    
loc_test = np.load(os.path.join('data', 'loc_test_springs' + str(n_balls) +\
                                 _s + str(test_size) + '.npy'))
edges_test = np.load(os.path.join('data', 'edges_test_springs' + str(n_balls) +\
                                 _s + str(test_size) + '.npy'))
vel_test = np.load(os.path.join('data', 'vel_test_springs' + str(n_balls) +\
                                 _s + str(test_size) + '.npy'))
    
energy_train = _energy(loc_train, vel_train, edges_train, interaction_strength)
energy_valid = _energy(loc_valid, vel_valid, edges_valid, interaction_strength)
energy_test = _energy(loc_test, vel_test, edges_test, interaction_strength)
    
plt.hist(energy_train.reshape(-1), bins=50, edgecolor='k', alpha=0.7)
print(np.mean(energy_train.reshape(-1)))
print(np.std(energy_train.reshape(-1)))

thres = 0.3 #around mean + std


    
    
    
    
    
    
    
    
