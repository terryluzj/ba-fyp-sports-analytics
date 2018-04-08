import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..\..')))

from analysis.predictive.settings import PRED_FILE_DIRECTORY, DEPENDENT_COLUMNS_FEATURED, DEPENDENT_COLUMNS
from analysis.predictive.rnn_model.pipeline import load_data
from analysis.predictive.rnn_model.settings import DATA_DIRECTORY, logger

MEGA_FEATURE_DIRECTORY = '{}meta_feature\\'.format(DATA_DIRECTORY)

if __name__ == '__main__':
    # Get featured data originally in rnn data file
    original_train, original_test, original_validation = load_data()
    dependent = list(map(lambda column_name: 'y_{}'.format(column_name), DEPENDENT_COLUMNS)) + ['last_run_time']

    # Retrieve meta features and store new file in the meta-feature data folder
    for column in DEPENDENT_COLUMNS_FEATURED:
        # Read in target column mega feature data (originally only in tran and test)
        train = pd.read_csv(PRED_FILE_DIRECTORY + 'meta_tuned\\%s.csv' % column)
        test = pd.read_csv(PRED_FILE_DIRECTORY + 'meta_tuned_test\\/%s.csv' % column)
        logger.warning('Retrieving meta-feature for {}...'.format(column))
        train['run_date'] = train['run_date'].apply(lambda value: pd.Timestamp(value))
        test['run_date'] = test['run_date'].apply(lambda value: pd.Timestamp(value))
        train.set_index(['horse_id', 'run_date'], inplace=True)
        test.set_index(['horse_id', 'run_date'], inplace=True)
        combined = train.append(test)

        # Conform to RNN data input as train, test and validation
        new_train = combined[combined.index.isin(original_train.index)].join(original_train[dependent], how='left')
        new_test = combined[combined.index.isin(original_test.index)].join(original_test[dependent], how='left')
        new_validation = combined[combined.index.isin(original_validation.index)].join(original_validation[dependent],
                                                                                       how='left')

        # Store as csv file
        logger.warning('Storing train, test and validation file for {}...'.format(column))
        new_train.to_csv('{}{}_meta_feature_training.csv'.format(MEGA_FEATURE_DIRECTORY, column))
        new_test.to_csv('{}{}_meta_feature_testing.csv'.format(MEGA_FEATURE_DIRECTORY, column))
        new_validation.to_csv('{}{}_meta_feature_validation.csv'.format(MEGA_FEATURE_DIRECTORY, column))
