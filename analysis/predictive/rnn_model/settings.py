import os

# Data related
FILE_DIRECTORY = '{}\\'.format(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')))
DATA_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'data')

# Feature and labels
TRAINING_LABEL = 'run_time_ma_window_3_diff'
