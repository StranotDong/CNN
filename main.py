import __future__
from models.hyper_parameters import Config

# Set all hyper parameters
config = Config()
hp = config.get_hyperparams();

print(hp)