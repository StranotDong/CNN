import __future__
import argparse as ap

"""
This file defines a class that can includes all hyper-parameters set by default 
or by command line.
"""

class Config:
	parser = ap.ArgumentParser()
		
	parser.add_argument('-lr', 
		'--learning_rate',
		type = float,
		default = 1e-3,
		help = 'learning rate')

	def get_hyperparams(self):
		return self.parser.parse_args()
