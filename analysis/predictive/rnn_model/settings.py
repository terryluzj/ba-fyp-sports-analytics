import logging
import os

# Data related
FILE_DIRECTORY = '{}\\'.format(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')))
DATA_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'data')

# Feature and labels
TRAINING_LABEL = 'y_{}'.format('run_time_1000')

# Logger
FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('logger')