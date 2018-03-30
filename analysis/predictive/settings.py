import os

FILE_DIRECTORY = '{}\\'.format(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '')))

REPORT_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'predictive\\report')
PRED_FILE_DIRECTORY = '{}predictive\\predictions\\'.format(FILE_DIRECTORY)

DATA_DIRECTORY = '{}{}\\'.format(FILE_DIRECTORY, 'data')
DATA_DIRECTORY_FEATURE_ENGINEERED = '{}\\{}\\'.format(DATA_DIRECTORY, 'feature_engineered')
DATA_DIRECTORY_PROCESSED = '{}\\{}\\'.format(DATA_DIRECTORY, 'processed')
