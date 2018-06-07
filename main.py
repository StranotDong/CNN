import __future__

import tensorflow as tf
import numpy as np

from models.configuration import Config
from cs231n.data_utils import load_CIFAR10
from models.CNN import basic_model
import matplotlib.pyplot as plt






# Set all hyper parameters
config = Config('config.json')
# print(config.learning_rate)


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
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print('Train data shape: ', X_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)

model = basic_model(config)
model_path = 'trained_models/basic_model/test.ckpt'
# track_losses = True
model_path = model.train(
	X_train, y_train, 
	X_val, y_val, 
	result_dir=model_path, 
	)

# if(track_losses):
# 	# plot losses		
# 	# plt.plot(losses)
# 	# plt.grid(True)
# 	# plt.title('Epoch 1 Loss')
# 	# plt.xlabel('minibatch number')
# 	# plt.ylabel('minibatch loss')
# 	# plt.show()

# 	# use tensorboard to plot
# 	## time stamp to name the file we want to store
# 	now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# 	root_logdir = "tf_logs"
# 	logdir = "{}/run-{}/".format(root_logdir, now)



"""
1. estimator
2. tensor board
3. verbosity
4. cpu, gpu
"""