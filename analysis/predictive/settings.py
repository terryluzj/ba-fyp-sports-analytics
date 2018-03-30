import os

# File path related
FILE_DIRECTORY = '{}\\'.format(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '')))

REPORT_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'predictive\\report')
PRED_FILE_DIRECTORY = '{}predictive\\predictions\\'.format(FILE_DIRECTORY)

DATA_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'data')
DATA_DIRECTORY_FEATURE_ENGINEERED = '{}\\{}\\'.format(DATA_DIRECTORY, 'feature_engineered')
DATA_DIRECTORY_PROCESSED = '{}\\{}\\'.format(DATA_DIRECTORY, 'processed')

# Features/labels related
TIME_WINDOW = 3
DEPENDENT_COLUMNS = ['run_time_1000', 'run_time_diff', 'run_time_quo', 'run_time_mean', 'run_time_median'] + \
                    ['run_time_ma_window_%s' % str(idx) for idx in range(2, TIME_WINDOW + 1)] + \
                    ['run_time_ewma_window_%s' % str(idx) for idx in range(2, TIME_WINDOW + 1)]
