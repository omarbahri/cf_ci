import os
import numpy as np
import tensorflow as tf
from cvae import CVAE  # Assuming cvae is a custom module where your CVAE model is defined

n_balls = 5

train_size = ''
valid_size = 200
test_size = 200

if train_size == '':
    _s = ''
else:
    _s = '_s' + str(train_size)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
    
results_path = os.path.join(os.sep, root_dir, 'shapelet_aug', 'results', 
                            'ci', 'particles_spring')

name = '_springs' + str(n_balls) + _s + '_uninfluenced2_oneconnect'

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), root_dir,
                        'Datasets', 'ci', 'particles_spring', name))
   
# Load X_test
X_train = np.load(os.path.join(data_path, 'X_train.npy'))
X_test = np.load(os.path.join(data_path, 'X_test.npy'))[:100]

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

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)

# Define the directory where the best model is saved
best_model_path = os.path.join(root_dir, 'Datasets', 'vaes', 'particles_spring',
                               name, 'models_42_0.0002_200', 'best_model.hdf5')

# Instantiate the CVAE model
latent_dim = 16  # Assuming you know the latent dimension

model = CVAE(latent_dim)

# Call the model to create its variables
_ = model(X_train[:1])  # Pass a sample through the model

# Load the weights of the best model
model.load_weights(best_model_path)

X_train = X_train[:100]

# Reconstruct X_test
reconstructed_X_train = model.decode(model.encode(X_train)[0]).numpy()

from matplotlib import pyplot as plt

X_train = X_train.numpy().reshape(100, 20, 49)
reconstructed_X_train = reconstructed_X_train.reshape(100, 20, 49)

plt.figure()
plt.plot(X_train[20,7], c='blue')
plt.plot(reconstructed_X_train[20,7], c='red')





