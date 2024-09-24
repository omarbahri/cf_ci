#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:35:28 2024

@author: omar
"""

import numpy as np
import os
import sys
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

# target_ball = int(sys.argv[1])
target_ball = 0

interaction_strength = 0.1

n_balls = 5

train_size = ''
valid_size = 200
test_size = 200

if train_size == '':
    _s = ''
else:
    _s = '_s' + str(train_size)
    
# name = '_springs' + str(n_balls) + _s + '_uninfluenced2_oneconnect'
name = 'uninfluenced2/_springs5_uninfluenced_s5000_oneconnect'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))    
output_dir = os.path.join(root_dir, 'cf_ci', 'scripts', 'data', name)

loc_train = np.load(os.path.join(output_dir, 'loc_train.npy'))
edges_train = np.load(os.path.join(output_dir, 'edges_train.npy'))
vel_train = np.load(os.path.join(output_dir, 'vel_train.npy'))
    
loc_valid = np.load(os.path.join(output_dir, 'loc_valid.npy'))
edges_valid = np.load(os.path.join(output_dir, 'edges_valid.npy'))
vel_valid = np.load(os.path.join(output_dir, 'vel_valid.npy'))
    
loc_test = np.load(os.path.join(output_dir, 'loc_test.npy'))
edges_test = np.load(os.path.join(output_dir, 'edges_test.npy'))
vel_test = np.load(os.path.join(output_dir, 'vel_test.npy'))
    
energy_train = _energy(loc_train, vel_train, edges_train, interaction_strength)
energy_valid = _energy(loc_valid, vel_valid, edges_valid, interaction_strength)
energy_test = _energy(loc_test, vel_test, edges_test, interaction_strength)
    
plt.hist(energy_train.reshape(-1), bins=50, edgecolor='k', alpha=0.7)
print(np.mean(energy_train.reshape(-1)))
print(np.std(energy_train.reshape(-1)))

thres = 0.3 #around mean + std

# Extract the final energies of the ball at index 0 for each simulation
final_energies_0_train = energy_train[:, -1, target_ball]  # Extract final energies for the ball at index target_ball
final_energies_0_valid = energy_valid[:, -1, target_ball]  # Extract final energies for the ball at index target_ball
final_energies_0_test = energy_test[:, -1, target_ball]  # Extract final energies for the ball at index target_ball

# Create the binary variable by comparing the final energies with the threshold
final_energy_0_0_train = final_energies_0_train > thres
final_energy_0_0_valid = final_energies_0_valid > thres
final_energy_0_0_test = final_energies_0_test > thres

# Convert boolean array to integer (0 or 1)
y_train = final_energy_0_0_train.astype(int)
y_valid = final_energy_0_0_valid.astype(int)
y_test = final_energy_0_0_test.astype(int)

vel_train = vel_train.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)
vel_valid = vel_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)
vel_test = vel_test.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)

# Reshape loc_train to shape (50000, 2 * 5, 50)
loc_train = loc_train.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)
loc_valid = loc_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)
loc_test = loc_test.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 50)

# Concatenate along the new feature axis (should be axis=1)
X_train = np.concatenate((loc_train, vel_train), axis=1)
X_valid = np.concatenate((loc_valid, vel_valid), axis=1)
X_test = np.concatenate((loc_test, vel_test), axis=1)

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..',
                                    'Datasets', 'ci', 'particles_spring', name))

# sys.exit()

if target_ball != 0:
    dataset_path = dataset_path + '_' + str(target_ball)
    
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

np.save(os.path.join(dataset_path, 'X_train.npy'), X_train)
np.save(os.path.join(dataset_path, 'y_train.npy'), y_train)
np.save(os.path.join(dataset_path, 'X_valid.npy'), X_valid)
np.save(os.path.join(dataset_path, 'y_valid.npy'), y_valid)
np.save(os.path.join(dataset_path, 'X_test.npy'), X_test)
np.save(os.path.join(dataset_path, 'y_test.npy'), y_test)
np.save(os.path.join(dataset_path, 'edges.npy'), edges_train[0])

sys.exit()

from sktime.transformations.panel.rocket import Rocket
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

clf = make_pipeline(
    Rocket(num_kernels=1000), RandomForestClassifier(n_estimators=1000))
    
X_all = np.concatenate((X_train, X_valid))
y_all = np.concatenate((y_train, y_valid))

clf.fit(X_all, y_all)
y_pred = clf.predict(X_test)

print(f1_score(y_pred, y_test))

    
    
    
