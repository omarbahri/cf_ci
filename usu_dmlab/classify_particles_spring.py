import numpy as np
import os
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime_convert import from_3d_numpy_to_nested
import sys

try:
    from sktime.transformations.panel.rocket import Rocket
    from sklearn.pipeline import make_pipeline
except:
    pass

seed = 42

random.seed(seed)
np.random.seed(seed)

name = sys.argv[1]
data_path = os.path.join(os.sep, 'data', 'Omar', 'cf_ci',
                         'data', 'particles_spring', name) 

X_train = np.load(os.path.join(data_path, 'X_train.npy'))
y_train = np.load(os.path.join(data_path, 'y_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))
X_valid = np.load(os.path.join(data_path, 'X_valid.npy'))
y_valid = np.load(os.path.join(data_path, 'y_valid.npy'))

X_train = np.concatenate([X_train[1000:], X_valid])
y_train = np.concatenate([y_train[1000:], y_valid])

X_test = np.concatenate([X_train[:1000], X_test])
y_test = np.concatenate([y_train[:1000:], y_test])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

X_train = from_3d_numpy_to_nested(X_train)
y_train = np.asarray(y_train)
X_test = from_3d_numpy_to_nested(X_test)
y_test = np.asarray(y_test)

clf = make_pipeline(
            Rocket(num_kernels=10, random_state=seed),
            RandomForestClassifier(n_estimators=100, random_state=seed))
        
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print('Rocket')
score = f1_score(y_pred_train, y_train, average='macro')
print('Train F1-score: ', score)
score = f1_score(y_pred_test, y_test, average='macro')
print('Test F1-score: ', score)



clf = KNeighborsTimeSeriesClassifier(distance="dtw")

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print('KNN TS')
score = f1_score(y_pred_train, y_train, average='macro')
print('Train F1-score: ', score)
score = f1_score(y_pred_test, y_test, average='macro')
print('Test F1-score: ', score)




clf = TimeSeriesForestClassifier(n_estimators=100, random_state=seed)

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print('TSF')
score = f1_score(y_pred_train, y_train, average='macro')
print('Train F1-score: ', score)
score = f1_score(y_pred_test, y_test, average='macro')
print('Test F1-score: ', score)