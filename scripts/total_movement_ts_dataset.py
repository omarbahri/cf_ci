#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:35:28 2024

@author: omar
"""

###

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

interaction_strength = 0.1

n_balls = 5

train_size = ''
valid_size = 200
test_size = 200

if train_size == '':
    _s = ''
else:
    _s = '_s' + str(train_size)
    
name = '_springs' + str(n_balls) + _s + '_uninfluenced_oneconnect'

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
    
diffs_train = np.diff(loc_train, axis=1)
distances_train = np.sqrt(np.sum(diffs_train**2, axis=2))
total_distances_train = np.sum(distances_train, axis=(1, 2))

diffs_valid = np.diff(loc_valid, axis=1)
distances_valid = np.sqrt(np.sum(diffs_valid**2, axis=2))
total_distances_valid = np.sum(distances_valid, axis=(1, 2))

diffs_test = np.diff(loc_test, axis=1)
distances_test = np.sqrt(np.sum(diffs_test**2, axis=2))
total_distances_test = np.sum(distances_test, axis=(1, 2))
    
plt.hist(total_distances_train, bins=50, edgecolor='k', alpha=0.7)
print(np.mean(total_distances_train.reshape(-1)))
print(np.std(total_distances_train.reshape(-1)))

thres = 13.0 #around mean + std

# Create the binary variable by comparing the final energies with the threshold
y_train = (total_distances_train >= thres).astype(int)
y_valid = (total_distances_valid >= thres).astype(int)
y_test = (total_distances_test >= thres).astype(int)

vel_train = vel_train.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)
vel_valid = vel_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)
vel_test = vel_test.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)

# Reshape loc_train to shape (50000, 2 * 5, 49)
loc_train = loc_train.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)
loc_valid = loc_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)
loc_test = loc_test.transpose(0, 3, 2, 1).reshape(-1, 2 * 5, 49)

# Concatenate along the new feature axis (should be axis=1)
X_train = np.concatenate((loc_train, vel_train), axis=1)
X_valid = np.concatenate((loc_valid, vel_valid), axis=1)
X_test = np.concatenate((loc_test, vel_test), axis=1)

# sys.exit()
    
name = name + '_total_movement'

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..',
                                    'Datasets', 'ci', 'particles_spring', name))
   
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

    
    
    
