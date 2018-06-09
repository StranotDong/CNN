
import __future__
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

"""
This file includes some cnn models
"""

"""
The basic model's structure is like this: conv_layers->fully_connected
conv_layers: cov->(batch_norm->)activation->max_pool
fully_connect: hidden_layers->output_layer
hidden_layers: linear->(batch_norm->)activation

Using exponential learning rate decay
"""
class basic_model:

	def __init__(self, config, rst_folder, model_name):
		# reset the graph
		tf.reset_default_graph()
		# get all hyper-parameters
		self.config = config
		# the file to save the final model
		self.rst_path = rst_folder + model_name
		if not os.path.exists(rst_folder):
			os.makedirs(rst_folder)
		

	'''
	This function constructs the convlutional layers
	Inputs: 
		inputs: the data fed in

	Returns:
		inputs: the inputs fed to the fully connected layers
	'''
	def conv_layers(self, inputs, is_training):
		num_layers = self.config.num_conv_layers
		kernel_sizes = self.config.conv_kernel_sizes
		pool_sizes = self.config.pool_sizes

		# init related
		init = None # default initializer is Xavier uniform
		init_type = self.config.conv_init.lower()
		if init_type == 'he':
			init = tf.contrib.layers.variance_scaling_initializer()

		with tf.name_scope('Conv_layers') as scope:
			for i in range(num_layers):
				# conv layer
				temp_out = tf.layers.conv2d(
					inputs=inputs, 
					filters=kernel_sizes[i][2], 
					kernel_size=kernel_sizes[i][0:2], 
					strides=self.config.conv_strides,
					padding=self.config.conv_paddings[i],
					name='conv'+str(i),
					kernel_initializer=init
					)

				# batch norm
				if self.config.batch_norm:
					temp_out = tf.layers.batch_normalization(
						temp_out, 
						training=is_training,
						epsilon=self.config.bn_epsilon,
						name='batch_norm'+str(i)
						)

				# activation
				conv_out = tf.nn.relu(temp_out, name='relu_in_conv'+str(i))

				# max pooling
				inputs = tf.layers.max_pooling2d(
					inputs=conv_out,
					pool_size=pool_sizes[i],
					strides=self.config.pool_strides[i],
					padding=self.config.pool_paddings[i],
					name='max_pooling'+str(i)
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
	def fully_connected(self, inputs, is_training):
		num_layers = self.config.num_hidden_layers
		layer_sizes = self.config.hidden_layer_sizes
		dense = inputs

		# init related
		init = None # default initializer is Xavier uniform
		init_type = self.config.fc_init.lower()	
		if init_type == 'he':
			init = tf.contrib.layers.variance_scaling_initializer()
					
		with tf.name_scope('Fully_connected') as scope:
			# hidden layers
			if num_layers != 0:
				for i in range(num_layers - 1):
					with tf.name_scope('Hidden') as scope1:
						## Linear
						temp_out = tf.layers.dense(
							dense, # input
							units=layer_sizes[i], 
							name='linear'+str(i),
							kernel_initializer=init
							)
						## batch normalization
						if self.config.batch_norm:
							temp_out = tf.layers.batch_normalization(
								temp_out, 
								training=is_training,
								epsilon=self.config.bn_epsilon,
								name='batch_norm'+str(i)
								)
						## activation
						dense = tf.nn.relu(temp_out, name='relu_in_fc'+str(i))

					# drop out #################################

			# output layer
			outputs = tf.layers.dense(
				dense, # input
				units=self.config.num_classes,
				name='Output',
				kernel_initializer=init
				)

		return outputs


	'''
	This function is to construct the basic_graph

	Returns:
		outputs: the data output from the baic_graph
	'''
	def basic_graph(self, inputs, is_training):		
		conv_out = self.conv_layers(inputs, is_training)
		flat_size = conv_out.shape[1]*conv_out.shape[2]*conv_out.shape[3]
		conv_flat = tf.reshape(conv_out, [-1, flat_size])
		outputs = self.fully_connected(conv_flat, is_training)

		return outputs

	'''
	This function is used to construct the total model
	Inputs:
		data_size: [data_length, data_width, data_depth]

	Returns:
		X: placeholder for batch data
		y: place holder for batch labels
		is_training: placeholder to determine whether this forward is to train the model
		correct_predictions: the number of correct predictions in one batch
		accuracy: the accuracy along one batch
		loss: total loss in one batch
		updates: backward propagation related
		learning_rate: current learning rate
	'''
	def model(self, data_size):
		# 1. Deciding the placeholders
		## input:
		data_length, data_width, data_depth = data_size # input size
		X = tf.placeholder(tf.float32, 
			[None, data_length, data_width, data_depth],name='X') # None to match the batch size

		## labels:
		y = tf.placeholder(tf.int64, [None],name='y')

		## if this data feeding is used for training (we need this info when doing batch normalization)
		is_training = tf.placeholder(tf.bool, name='is_training')


		# 2. constructing the basic graph
		logits = self.basic_graph(X, is_training)


		# 3. have tensorflow compute accuracy
		with tf.name_scope('Eval') as scope:
			correct_prediction = tf.equal(tf.argmax(logits,1), y, name='correct')
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

		# 4. back propagation: the loss and the optimizer
		## the loss
		with tf.name_scope('Loss') as scope:			
			if(self.config.loss == 'hinge'):
				# print(logits.shape)
				loss = tf.losses.hinge_loss(
					labels=tf.one_hot(y, self.config.num_classes), 
					logits=logits
					)
			elif(self.config.loss == 'softmax'):
				loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

		## global_step
		global_step = tf.Variable(0, trainable=False)

		## learning rate
		learning_rate = tf.train.exponential_decay(
			self.config.learning_rate, 
			global_step,
			self.config.decay_steps, 
			self.config.decay_rate
			)

		## the optimizer
		optimizer = tf.train.AdamOptimizer(
			learning_rate,
			beta1=self.config.adam_beta1,
			beta2=self.config.adam_beta2,
			epsilon=self.config.adam_epsilon
			)

		## extra update ops for batch normalization
		if self.config.batch_norm :
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(extra_update_ops):
				### minimizing loss after moving average and moving variance update in batch normalization
			    updates = optimizer.minimize(loss, global_step=global_step)
		## no batch normalization
		else:
			updates = optimizer.minimize(loss, global_step=global_step)

		placeholder_dict = {'X':X, 'y':y, 'is_training':is_training}
		updated_node_dict = {
			'correct_prediction':correct_prediction, 
			'accuracy': accuracy, 
			'loss':loss,
			'updates': updates,
			'global_step': global_step,
			'learning_rate': learning_rate
			}

		

		return placeholder_dict, updated_node_dict



	'''
	This function is used to train. The check points will be stored in ckpt folder if needed. It will
	do the validation after training if val_data and val_labels are not None.
	Inputs:
		data: total training data: shape is [number, length, width, depth]
		labels: total training labels
		val_data: validation data
		val_labels: validation labels

	Returns:
		result_path: the path that stores the trained model
	'''
	def train(
			self, 
			data, labels, 
			val_data=None, val_labels=None,  
		):
		 # shuffle indicies
		train_indicies = np.arange(data.shape[0])
		np.random.shuffle(train_indicies)

		# construct the model
		data_size = data.shape[1:]
		placeholder_dict, updated_node_dict = self.model(data_size)

		X = placeholder_dict['X'] 
		y = placeholder_dict['y']
		is_training = placeholder_dict['is_training']
		accuracy = updated_node_dict['accuracy'] 
		loss = updated_node_dict['loss']
		updates = updated_node_dict['updates']
		global_step = updated_node_dict['global_step']
		learning_rate = updated_node_dict['learning_rate']

		# the batch size
		batch_size = self.config.batch_size
		# the saver object for saving check points
		saver = tf.train.Saver()
	
		# whether print information in the training process
		is_print = self.config.print_every > 0
		# whether to track the one batch train stats
		is_batch_train_stats = self.config.batch_train_stats_every > 0
		# whether to track the whole train data stats
		is_whole_train_stats = self.config.whole_train_stats_every > 0
		# whether to do the validation in the process of training after certain epoches
		is_val = self.config.val_every > 0
		if is_val and (val_data is None or val_labels is None):
			print("Can't do the validation because no validation data is provided.")
		is_val = is_val and val_data is not None and val_labels is not None
		# whether to track learning rate in the training process
		is_lr = self.config.track_lr_every > 0
		# whether save check points in the process of training after certain epoches 
		is_ckpt = self.config.ckpt_every > 0	


		now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
		root_logdir = "tf_logs" # the root dir that stores tensor board files
		# keep track of train stats and validation stats and put them to tensor board		
		if(is_batch_train_stats or is_whole_train_stats or is_val):	
			acc_summary = tf.summary.scalar('ACCURACY', accuracy)
			loss_summary = tf.summary.scalar('LOSS', loss)
			merged = tf.summary.merge_all()			

		# batch training stats tensor board file
		if(is_batch_train_stats):			
			logdir = "{}/batch_train_stats-{}/".format(root_logdir, now)
			batch_train_summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

		# whole train data stats tensor board file
		if(is_whole_train_stats):			
			logdir = "{}/whole_train_stats-{}/".format(root_logdir, now)
			whole_train_summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

		# validation stats tensor board file
		if(is_val):			
			logdir = "{}/val_stats-{}/".format(root_logdir, now)
			val_summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

		# keep track of learning rate
		if is_lr:
			lr_summary = tf.summary.scalar('Learning rate', learning_rate)
			logdir = "{}/lr_stats-{}/".format(root_logdir, now)
			lr_summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

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
			    		y: val_labels,
			    		is_training: False
			    	}
			## for training set evaluation
			whole_train_eval_dict = {
				X: data,
			    y: labels,
			    is_training: False
			}

			# training starts
			print("Training")
			step = 0
			for e in range(self.config.epoch):
				# make sure we iterate over the dataset once
				for i in range(int(math.ceil(data.shape[0]/batch_size))):
					# generate indicies for the batch
					start_idx = (i*batch_size)%data.shape[0]
					idx = train_indicies[start_idx:start_idx+batch_size]

					# construct the feed dict
					feed_dict = {
						X: data[idx,:],
						y: labels[idx],
						is_training: True
					}	            						

					# track the batch train stats
					if(is_batch_train_stats and step % self.config.batch_train_stats_every == 0):						
						summary = sess.run(merged, feed_dict=feed_dict)
						batch_train_summary_writer.add_summary(summary, step)

					# track the whole train data stats
					if(is_whole_train_stats and step % self.config.whole_train_stats_every == 0):						
						summary = sess.run(merged, feed_dict=whole_train_eval_dict)
						whole_train_summary_writer.add_summary(summary, step)

					# track the validation stats
					if(is_val and step % self.config.val_every == 0):	            	
						# val_loss, val_accuracy = sess.run(val_fetch, feed_dict=val_dict)
						# print("Iteration {0}, Validation loss = {1:.3g} and accuracy of {2:.3g}"\
						#       .format(step, val_loss,val_accuracy))
						summary = sess.run(merged, feed_dict=val_dict)
						val_summary_writer.add_summary(summary, step)

					# track the learning rate
					# track the whole train data stats
					if(is_lr and step % self.config.track_lr_every == 0):						
						summary = lr_summary.eval(feed_dict=feed_dict)
						lr_summary_writer.add_summary(summary, step)

					# run
					batch_loss, _, batch_accuracy, step = sess.run(fetch, feed_dict=feed_dict)
					iter_cnt = step - 1				
					
					# print stats for every n iteration
					if(is_print and iter_cnt % self.config.print_every == 0):						
						print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"
					  .format(iter_cnt, batch_loss, batch_accuracy))

					# save check points after n epoches
					if not os.path.exists('ckpts'):
						os.makedirs('ckpts')
					if(is_ckpt and iter_cnt % self.config.ckpt_every == 0):	            	
						ckpt_path = saver.save(sess, "ckpts/model_iter"+str(iter_cnt)+".ckpt")

			# validation after training
			if(val_data is not None and val_labels is not None):
				val_loss, val_accuracy = sess.run(val_fetch, feed_dict=val_dict)
				print("\nValidation\nValidation loss = {0:.3g} and accuracy of {1:.3g}"\
				      .format(val_loss,val_accuracy))

			# close the tensorboard file writer
			if is_val:
				summary = sess.run(merged, feed_dict=val_dict)
				val_summary_writer.add_summary(summary, step)
				val_summary_writer.close()
			if is_batch_train_stats:
				batch_train_summary_writer.close()
			if is_whole_train_stats:
				whole_train_summary_writer.close()
				

			# save the final results
			saver.save(sess, self.rst_path)



	'''
	This function is used for prediction. Can predict multiple data at the same time
	The shape of the data is (batch_size, length, width, depth)
	'''           	
	def predict(self, data):	
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph(self.rst_path+'.meta')
			saver.restore(sess, self.rst_path)
			graph = tf.get_default_graph()
			X = graph.get_tensor_by_name("X:0")
			is_training = graph.get_tensor_by_name("is_training:0")
			logits = graph.get_tensor_by_name("Fully_connected/Output/BiasAdd:0")

			one_hot_rst = logits.eval({X:data, is_training:False})

		return np.argmax(one_hot_rst,1)

	'''
	This function is used for evaluation
	'''           	
	def evaluate(self, data, labels):
		
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph(self.rst_path+'.meta')
			saver.restore(sess, self.rst_path)
			graph = tf.get_default_graph()
			X = graph.get_tensor_by_name("X:0")
			y = graph.get_tensor_by_name("y:0")
			is_training = graph.get_tensor_by_name("is_training:0")
			accuracy = graph.get_tensor_by_name("Eval/accuracy:0")

			rst = accuracy.eval({X:data, y:labels, is_training:False})

		return rst

		            	





