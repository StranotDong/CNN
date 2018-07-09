import __future__

import json
import os

"""
The function that generates the configuration json file
Inputs:
	folder+filename: the path to save the json file
	configs: the set configurations
Outputs:
	config_dict: the configuration dictionary
"""
def config_generator(folder, filename, **configs):
	if not os.path.exists(folder):
		os.makedirs(folder)

	config_dict = {}
	for key, value in configs.items():
		config_dict[key] = value

	with open(folder+filename, 'w') as f:
		json.dump(config_dict, f)

	return config_dict


folder = 'config_files/transfer_learning/'
filename = 'test.json'

config_dict = config_generator(
	folder,
	filename,
	epoch=1,
	val_every=500,
	batch_train_stats_every=1,
	batch_norm=True,
	dropout=True,
	num_hidden_layers=0,
	hidden_layer_sizes=[]
)