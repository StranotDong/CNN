
import __future__
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

"""
This file includes some cnn models
"""

"""
The basic model's structure is like this: conv_layers->fully_connected
conv_layers: cov->RELU->max_pool->...->fully_connected
fully_connect: hidden_layers->output_layer
"""
class basic_model:

	def __init__(self, config):
		# get all hyper-parameters
		self.config = config


	'''
	This function constructs the convlutional layers
	Inputs: 
		inputs: the data fed in

	Returns:
		inputs: the inputs fed to the fully connected layers
	'''
	def conv_layers(self, inputs):
		num_layers = self.config.num_conv_layers
		kernel_sizes = self.config.conv_kernel_sizes
		pool_sizes = self.config.pool_sizes

		for i in range(num_layers):
			# conv layer
			conv_out = tf.layers.conv2d(
				inputs=inputs, 
				filters=kernel_sizes[i][2], 
				kernel_size=kernel_sizes[i][0:2], 
				strides=self.config.conv_strides,
				padding=self.config.conv_paddings[i],
				activation=tf.nn.relu
				# initialization ###############################
				)

			# max pooling
			inputs = tf.layers.max_pooling2d(
				inputs=conv_out,
				pool_size=pool_sizes[i],
				strides=self.config.pool_strides[i],
				padding=self.config.pool_paddings[i]
				)

			# batch normalization ########################################			

		return inputs


	'''
	The function used to construct the graph of the fully connected layers
	Inputs:
		inputs: 1 X flatted_conv_layer_output_dimension
		W: num_layers X flatted_conv_layer_output_dimension X this_layer_output_dimension

	Returns:
		outputs: 1 X num_classes
	'''
	def fully_connected(self, inputs):
		num_layers = self.config.num_hidden_layers
		layer_sizes = self.config.hidden_layer_sizes
		dense = inputs

		# hidden layers
		if num_layers != 0:
			for i in range(num_layers - 1):
				dense = tf.layers.dense(
					dense, # input
					units=layer_sizes[i], 
					activation=tf.nn.relu
					# initialization ####################
					)
				# drop out #################################

		# output layer
		outputs = tf.layers.dense(
			dense, # input
			units=self.config.num_classes,
			# initialization ####################
			)

		return outputs


	'''
	This function is to construct the basic_graph

	Returns:
		outputs: the data output from the baic_graph
	'''
	def basic_graph(self, inputs):		
		conv_out = self.conv_layers(inputs)
		flat_size = conv_out.shape[1]*conv_out.shape[2]*conv_out.shape[3]
		conv_flat = tf.reshape(conv_out, [-1, flat_size])
		outputs = self.fully_connected(conv_flat)

		return outputs

	'''
	This function is used to construct the total model
	Inputs:
		data_size: [data_length, data_width, data_depth]

	Returns:
		X: placeholder for batch data
		y: place holder for batch labels
		correct_predictions: the number of correct predictions in one batch
		accuracy: the accuracy along one batch
		loss: total loss in one batch
		updates: backward propagation related
	'''
	def model(self, data_size):
		# 1. Deciding the placeholders
		## input:
		data_length, data_width, data_depth = data_size # input size
		X = tf.placeholder(tf.float32, 
			[None, data_length, data_width, data_depth]) # None to match the batch size

		## labels:
		y = tf.placeholder(tf.int64, [None])

		# ## if this data feeding is used for training
		# is_training = tf.placeholer(tf.bool)


		# 2. constructing the basic graph
		logits = self.basic_graph(X)


		# 3. have tensorflow compute accuracy
		correct_prediction = tf.equal(tf.argmax(logits,1), y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


		# 4. back propagation: the loss and the optimizer
		## the loss
		if(self.config.loss == 'hinge'):
			# print(logits.shape)
			loss = tf.losses.hinge_loss(
				labels=tf.one_hot(y, self.config.num_classes), 
				logits=logits
				)
		elif(self.config.loss == 'softmax'):
			loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

		global_step = tf.Variable(0, trainable=False)
		## the optimizer
		optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
		updates = optimizer.minimize(loss, global_step=global_step)

		return X, y, correct_prediction, accuracy, loss, updates, global_step



	'''
	This function is used to train
	Inputs:
		data: total training data: shape is [number, length, width, depth]
		labels: total training labels
		val_data: validation data
		val_labels: validation labels
		result_dir: the directory for saving the trained model
		track_losses: whether to track the first epoch losses

	Returns:
		result_path: the path that stores the trained model
		losses: the last epoch losses tracked while training
	'''
	def train(self, data, labels, val_data, val_labels, result_dir = 'temp/final_model.ckpt', track_losses=False):
		 # shuffle indicies
		train_indicies = np.arange(data.shape[0])
		np.random.shuffle(train_indicies)

		# construct the model
		data_size = data.shape[1:]
		X, y, _, accuracy, loss, updates, global_step = self.model(data_size)

		batch_size = self.config.batch_size
		saver = tf.train.Saver()

		# run train process
		with tf.Session() as sess:

			# initialize
			sess.run(tf.global_variables_initializer())
			
			# decide the fetch
			## for train
			fetch = [loss, updates, accuracy, global_step] 
			## for validation
			val_fetch = [loss, accuracy] 
			val_dict = {
			    		X: val_data,
			    		y: val_labels
			    	}

			# training starts
			print("Training")
			# keep track of losses for the first epoch if needed
			losses = []
			for e in range(self.config.epoch):
				# make sure we iterate over the dataset once
				for i in range(int(math.ceil(data.shape[0]/batch_size))):
					# generate indicies for the batch
					start_idx = (i*batch_size)%data.shape[0]
					idx = train_indicies[start_idx:start_idx+batch_size]

					# construct the feed dict
					feed_dict = {
						X: data[idx,:],
						y: labels[idx]
					}	            	

					# run
					loss, _, accuracy, global_step = sess.run(fetch, feed_dict=feed_dict)

					##### tensor board ##################
					if(track_losses and e == 0):
						losses.append(loss)

					# print stats for every n iteration
					# print(global_step)
					if(self.config.print_every > 0 and global_step % self.config.print_every == 0):
						print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"
					  .format(global_step, loss, accuracy))

				# validation after n epoches
				if(self.config.val_every > 0 and (e+1) % self.config.val_every == 0):	            	
					val_loss, val_accuracy = sess.run(val_fetch, feed_dict=val_dict)
					print("Epoch {0}, Validation loss = {1:.3g} and accuracy of {2:.3g}"\
					      .format(e+1, val_loss,val_accuracy))

				# save check points after n epoches
				if(self.config.ckpt_every > 0 and (e+1) % self.config.ckpt_every == 0):	            	
					ckpt_path = saver.save(sess, "temp/my_model_epoch"+str(e+1)+".ckpt")

			# validation after training
			val_loss, val_accuracy = sess.run(val_fetch, feed_dict=val_dict)
			print("\nValidation\nValidation loss = {1:.3g} and accuracy of {2:.3g}"\
			      .format(e+1, val_loss,val_accuracy))

			# save the final results
			result_path = saver.save(sess, result_dir)

		if(track_losses):
			return result_path, losses

		return result_path, None
		            	
		            
		            	





