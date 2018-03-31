import logging
import pandas as pd
# import tensorflow as tf
from analysis.predictive.feature_engineering import feature_engineer, drop_cols
from analysis.predictive.settings import DATA_DIRECTORY_FEATURE_ENGINEERED, DATA_DIRECTORY_PROCESSED, DEPENDENT_COLUMNS
from analysis.predictive.rnn_model.settings import TRAINING_LABEL

FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('logger')


def read_feature_engineered_dataframe(drop_rank_info=True, filter_columns=True,
                                      reuse=True, reuse_name=None, **params):
    # Read from feature engineered dataframe from file named df_combined_all.csv
    if reuse:
        logger.warning('Reusing stored feature engineered file %s.csv...' % reuse_name)
        race_combined_featured = pd.read_csv('{}{}.csv'.format(DATA_DIRECTORY_FEATURE_ENGINEERED, reuse_name),
                                             index_col=0)

        # Do type casting and multi-indexing
        race_combined_featured['run_date'] = race_combined_featured['run_date'].apply(lambda x: pd.Timestamp(x))
        race_combined_featured.set_index(['horse_id', 'run_date'], inplace=True)
    else:
        # Do feature engineering on the dataframe if specified (pass params for options defined in the function)
        logger.warning('Starting feature engineering on dataframe through read_race_dataframe')
        race_combined_featured_tuple = feature_engineer(read_race_dataframe(**params, reset_index=True),
                                                        df_name='df_combined_all_new')

        # As feature engineer function returns tuple of dataframe and rank info, do necessary assignment
        race_combined_featured = race_combined_featured_tuple[0]
        if not drop_rank_info:
            race_combined_featured['finishing_position'] = race_combined_featured_tuple[1]

    if reuse and drop_rank_info:
        # Drop ranking info when reusing the stored feature-engineered file
        race_combined_featured.drop('finishing_position', axis=1, inplace=True)

    # Remove dependent columns except the training column if specified
    logger.warning('Returning the feature engineered dataframe %s...' % str(race_combined_featured.shape))
    return remove_dependent_columns(race_combined_featured) if filter_columns else race_combined_featured


def read_race_dataframe(include_first_occurrence=False, filter_columns=True, reset_index=True):
    # Read raw dataframe from file horse_race_combined.csv
    logger.warning('Loading horse_race_combined.csv file...')
    race_combined = pd.read_csv('{}{}'.format(DATA_DIRECTORY_PROCESSED, 'horse_race_combined.csv'),
                                index_col=0)

    # Do type casting, sorting and multi-indexing
    logger.warning('Loaded horse_race_combined.csv successfully. Doing type casting and multi-indexing...')
    race_combined['run_date'] = race_combined['run_date'].apply(lambda x: pd.Timestamp(x))
    race_combined = race_combined.sort_values(['horse_id', 'run_date'])
    race_combined.set_index(['horse_id', 'run_date'], inplace=True)

    # Read data from those with first occurrence and do type casting and multi-indexing
    logger.warning('Loading first_occurrence_race.csv file...')
    first_occurrence_df = pd.read_csv('{}{}'.format(DATA_DIRECTORY_PROCESSED, 'first_occurrence_race.csv'),
                                      index_col=0)
    logger.warning('Loaded first_occurrence_race.csv successfully. Doing type casting and multi-indexing...')
    first_occurrence_df['run_date'] = first_occurrence_df['run_date'].apply(pd.Timestamp)

    # Drop some unwanted columns and sort accordingly
    drop_cols(first_occurrence_df)
    first_occurrence_df = first_occurrence_df.sort_values(['horse_id', 'run_date'])
    first_occurrence_df.set_index(['horse_id', 'run_date'], inplace=True)
    first_occurrence_df_index = first_occurrence_df.index

    # Filter current dataframe by removing those without prior time information
    logger.warning('Filtering out first occurrence records...')
    race_combined = race_combined[~race_combined.index.isin(first_occurrence_df_index)]

    # If the returned dataframe should include those with very first race record
    if include_first_occurrence:
        logger.warning('Include first occurrence data as include_first_occurrence is %s' % include_first_occurrence)
        race_combined = race_combined.append(first_occurrence_df)
        race_combined.sort_index(level=[0, 1], inplace=True)

    if filter_columns:
        # Remove dependent columns except the training column if specified
        logger.warning('Removing redundant columns...')
        remove_dependent_columns(race_combined)

    # Reset index by default for feature engineering
    logger.warning('Returning the dataframe %s resetting index...' % ('with' if reset_index else 'without'))
    return race_combined.reset_index() if reset_index else race_combined


def remove_dependent_columns(df):
    # Remove dependent-variable feature engineered columns not intended for training
    columns_to_drop = list(filter(lambda label: label != TRAINING_LABEL, DEPENDENT_COLUMNS))
    columns_dropped = []

    # Safely drop columns from the filtered list
    for col_name in columns_to_drop:
        try:
            # Add column name to the new list if successfully dropped
            df.drop(col_name, axis=1, inplace=True)
            columns_dropped.append(col_name)
        except ValueError:
            continue

    # Return the dataframe after dropping columns
    return df


def train_test_split_by_date(df):
    pass


if __name__ == '__main__':
    # Testing
    # df = read_race_dataframe(include_first_occurrence=True, reset_index=False)
    # feature_df = read_feature_engineered_dataframe(reuse=True, reuse_name='df_combined_all_new')
    pass
