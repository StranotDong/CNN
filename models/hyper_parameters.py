import __future__
# import argparse as ap
import json
import sys

"""
This file defines a class that can includes all hyper-parameters from
a json file
"""
class Config:

	'''
	Inputs: 
		file: a json file contains all hyperparameters
	'''
	def __init__(self, filename):
		# should first go to the dir where includes main.py
		with open(filename) as f:
			dat = f.read()
		hp = json.loads(dat)

		# learning rate
		self.learning_rate = hp.get('learning_rate', 1e-3)

		# number of classes
		self.num_classes = hp.get('num_classes', 10)

		# epoch
		self.epoch = hp.get('epoch', 1)

		# batch size
		self.batch_size = hp.get('batch_size', 64)

		# print stats for every n iters, 0 for not print
		self.print_every = hp.get('batch_size', 100)

		# do the validation for every n epoches, 0 for not doing
		self.val_every = hp.get('val_every', 0)

		# save check points for every n epoches, 0 for not saving
		self.ckpt_every = hp.get('ckpt_every', 0)



		# number of convolutional layers
		self.num_conv_layers = hp.get('num_conv_layers', 2)

		# convolutional kernel sizes: 
		# [kernel_length, kernel_width, num_filters]
		self.conv_kernel_sizes = hp.get('conv_kernel_sizes', [[5,5,32],[5,5,64]])

		if self.num_conv_layers != len(self.conv_kernel_sizes):
			sys.exit('The number of conv layers mismatches between num_conv_layers and conv_kernel_sizes')

		if self.num_conv_layers != 0 and len(self.conv_kernel_sizes[0]) != 3:
			sys.exit("num_conv_layers: format error")

		# stride for each conv_layer
		self.conv_strides = hp.get('strides', [1,1])

		if self.num_conv_layers != len(self.conv_strides):
			sys.exit('The number of conv layers mismatches between num_conv_layers and conv_strides')

		# conv padding type (valid or same)
		self.conv_paddings = hp.get('conv_padding', ['same', 'same'])

		if self.num_conv_layers != len(self.conv_paddings):
			sys.exit('The number of conv layers mismatches between num_conv_layers and conv_paddings')



		# pool sizes
		self.pool_sizes = hp.get('pool_sizes', [[2,2],[2,2]])
		
		if self.num_conv_layers != len(self.pool_sizes):
			sys.exit('The number of conv layers mismatches between num_conv_layers and pool_sizes')

		if self.num_conv_layers != 0 and len(self.pool_sizes[0]) != 2:
			sys.exit('strides: format error')

		# pool strides
		self.pool_strides = hp.get('pool_strides', [2,2])

		if self.num_conv_layers != len(self.pool_strides):
			sys.exit('The number of conv layers mismatches between num_conv_layers and pool_strides')

		# pool padding type (valid or same)
		self.pool_paddings = hp.get('pool_padding', ['valid', 'valid'])
		
		if self.num_conv_layers != len(self.pool_paddings):
			sys.exit('The number of conv layers mismatches between num_conv_layers and pool_paddings')



		# number of hidden layers in fully connected layers
		self.num_hidden_layers = hp.get('num_hidden_layers', 1)

		# the size of hidden layers in fully connected layers 
		self.hidden_layer_sizes = hp.get('hidden_layer_sizes', [1024])

		if self.num_hidden_layers != len(self.hidden_layer_sizes):
			sys.exit('The number of hidden layers mismatches between num_hidden_layers and hidden_layer_sizes')



		# loss type (hinge or softmax)
		self.loss = hp.get('loss', 'hinge')
		if self.loss != 'hinge' and self.loss != 'softmax':
			sys.exit('loss should be hinge or softmax')

# class Config:
# 	parser = ap.ArgumentParser()
		
# 	parser.add_argument(
# 		'-lr', 
# 		'--learning_rate',
# 		type = float,
# 		default = 1e-3,
# 		help = 'learning rate'
# 		)

# 	parser.add_argument(
# 		'-nconv',
# 		'--num_conv_layers',
# 		type = int,
# 		default = 2,
# 		help = '# of convolutional layers'
# 		)

# 	# parser.add_argument(
# 	# 	'-kz',
# 	# 	'--conv_kernel_sizes',
# 	# 	type = int,
# 	# 	default = [[5,5,32],[5,5,64]], 
# 	# 	help = 'convolutional kernel sizes: [kernel_length, kernel_width, num_filters]'
# 	# 	)

# 	parser.add_argument(
# 		'-nh',
# 		'--num_hidden_layers',
# 		type = int,
# 		default = 2,
# 		help = '# of hidden layers of the fully connected layers'
# 		)

# 	# parser.add_argument(
# 	# 	'-hz',
# 	# 	'--hidden_layer_sizes',
# 	# 	type = int,
# 	# 	default = [1024,100],
# 	# 	help = 'an array, the size of hidden layers of the fully connected layers'
# 	# 	)

# 	parser.add_argument(
# 		'-ncl', 
# 		'--num_classes',
# 		type = int,
# 		default = 10,
# 		help = '# of classes'
# 		)

# 	def __init__(self):
# 		self.__hidden_layer_sizes()

# 	def __hidden_layer_sizes(self):
# 		hp = self.parser.parse_args()
# 		num_layers = hp.num_hidden_layers
# 		for i in range(num_layers):
# 			self.parser.add_argument(
# 				'-hz' + str(i+1),
# 				'--hidden_layer' + str(i+1) + '_sizes',
# 				type = int,
# 				default = 1024,
# 				help = 'the size of hidden_layer' + str(i+1) + ' in the fully connected layers'
# 				)
# 		# self.parser.add_argument(hi)


# 	def get_hyperparams(self):
# 		hp = self.parser.parse_args()
# 		num_layers = hp.num_hidden_layers
# 		hp.hidden_layer_sizes = []
# 		for i in range(num_layers):
# 			hp.hidden_layer_sizes.add(hp)
# 		return 
