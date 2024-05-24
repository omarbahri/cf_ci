#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:12:08 2022

@author: omar
"""
#0.081817454676598

import os
import numpy as np
import sys
import sklearn
from classifiers import resnet_val as resnet
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import random
import csv
import fcntl
import pandas as pd

# def fit_transform_classifier(x_train, y_train, x_test, y_test, output_directory,
#                      nb_epochs, weights_directory=None):
#     nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

#     # transform the labels from integers to one hot vectors
#     enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
#     enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
#     y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
#     y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

#     if len(x_train.shape) == 2:  # if univariate
#         # add a dimension to make it multivariate with one dimension 
#         x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
#         x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

#     input_shape = x_train.shape[1:]
#     classifier = resnet.Classifier_RESNET(output_directory,
#                             input_shape,
#                             nb_classes, verbose=True,
#                             load_weights= False)
    
#     classifier.fit(X_train, y_train, nb_epochs)
    
#     y_pred = classifier.my_predict(X_test)
#     return y_pred
    
results_path = os.path.join(os.sep, 'data', 'Omar', 'shapelet_aug', 'results', 
                            'ci', 'particles_spring')

name = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
nb_epochs = int(sys.argv[4])

data_path = os.path.join(os.sep, 'data', 'Omar', 'particles_spring',
                         'data', name) 

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

output_csv_path = os.path.join(results_path, 
           'particles_spring_resnetval_f1macro_' + str(seed) + '.csv')
    
try:    
    output_csv = pd.read_csv(output_csv_path, header=None)
except Exception:
    print('Gonna create this file later')
    
try:
    if name in list(output_csv[output_csv.columns[0]]):
        sys.exit()
except Exception:
    print('')
    
# nb_epochs = 1500

print("Loaded Dataset.."+str(name))

X_train = np.load(os.path.join(data_path, 'X_train.npy'))
y_train = np.load(os.path.join(data_path, 'y_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))
X_valid = np.load(os.path.join(data_path, 'X_valid.npy'))
y_valid = np.load(os.path.join(data_path, 'y_valid.npy'))

X_train = np.concatenate([X_train, X_valid])
y_train = np.concatenate([y_train, y_valid])

output_directory = os.path.join(results_path, name, 'resnetval',
                'models_' + str(seed) + '_' + str(lr) + '_' + str(nb_epochs))

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], -1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2], -1))

if len(X_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
p = np.random.RandomState(seed=42).permutation(len(y_train))
X_train, y_train = X_train[p], y_train[p]
     
name_csv = '_'.join([name, str(seed), str(lr), str(nb_epochs)])                   

# y_pred = fit_transform_classifier(X_train, y_train, X_test, y_test,
#                  output_directory, nb_epochs)

score = f1_score(y_pred, y_test, average='macro')
print('F1-score: ', score)

with open(output_csv_path, 'a', newline='') as fw:
    fcntl.flock(fw, fcntl.LOCK_EX)
    writer = csv.writer(fw)
    writer.writerow([name_csv + '__final', score])
    fcntl.flock(fw, fcntl.LOCK_UN)
