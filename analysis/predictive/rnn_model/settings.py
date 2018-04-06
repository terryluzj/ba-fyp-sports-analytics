import logging
import os

# Data related
FILE_DIRECTORY = '{}\\'.format(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')))
DATA_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'data')

# Feature and labels
TRAINING_LABEL = 'y_{}'.format('run_time_diff')
PREDICT_FIRST_RACE = False

# Logging
FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('logger')


def get_current_training_process(percentage):
    # Get string form of current training process
    if percentage >= 97.5:
        percentage = 100
    elif percentage <= 5:
        percentage = 5
    num_block = percentage // 5
    return '[{}]'.format(''.join(['=' * (num_block - 1)] + ['>'] + ['-' * (20 - num_block)]))
