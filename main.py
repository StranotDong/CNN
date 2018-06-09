import __future__

import tensorflow as tf
import numpy as np

from models.configuration import Config
from cs231n.data_utils import load_CIFAR10
from models.CNN import basic_model
import matplotlib.pyplot as plt


# Get the data
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(num_training=1000,num_validation=10, num_test=10)
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print('Train data shape: ', X_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)


# Set all hyper parameters
models = []
for i in range(1):
    config = Config('config_files/config_lr{0:s}.json'.format(str(i)))
    model_folder = 'trained_models/basic_model'
    filename = 'test{0:s}.ckpt'.format(str(i))

    models.append(basic_model(config, model_folder, filename))
    
    model_path = models[i].train(
    	X_train, y_train, 
    	X_val, y_val, 
    	)

# for i in range(2):
#     acc = models[i].evaluate(X_val, y_val)
#     print(acc)

#     pred = models[i].predict(X_val)
#     print(pred)



"""
1 regularization
2 grid search, random search
3. transfer learning
4. cpu, gpu, multithread
"""