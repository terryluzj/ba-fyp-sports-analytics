import pandas as pd
import tensorflow as tf
from analysis.predictive.feature_engineering import feature_engineer
from analysis.predictive.settings import DATA_DIRECTORY_FEATURE_ENGINEERED, DATA_DIRECTORY_PROCESSED, DEPENDENT_COLUMNS
from analysis.predictive.rnn_model.settings import TRAINING_LABEL


def read_feature_engineered_dataframe(drop_rank_info=True, filter_columns=True, reuse=True, **params):
    # Read from file df_combined_all.csv
    if reuse:
        race_combined_featured = pd.read_csv('{}{}'.format(DATA_DIRECTORY_FEATURE_ENGINEERED, 'df_combined_all.csv'), index_col=0)
        race_combined_featured['run_date'] = race_combined_featured['run_date'].apply(lambda x: pd.Timestamp(x))
        race_combined_featured.set_index(['horse_id', 'run_date'], inplace=True)
    else:
        # Do feature engineering on the dataframe if specified (pass params for options defined in the function)
        race_combined_featured_tuple = feature_engineer(read_race_dataframe(**params, reset_index=True), df_name='df_combined_all_new')
        race_combined_featured = race_combined_featured_tuple[0]
        if not drop_rank_info:
            race_combined_featured['finishing_position'] = race_combined_featured_tuple[1]

    if not reuse and drop_rank_info:
        # Drop ranking info when reusing the stored feature-engineered file
        race_combined_featured.drop('finishing_position', axis=1, inplace=True)
    return remove_dependent_columns(race_combined_featured) if filter_columns else race_combined_featured


def read_race_dataframe(filter_first_occurrence=True, filter_columns=True, reset_index=True):
    # Read from file horse_race_combined.csv
    race_combined = pd.read_csv('{]{}'.format(DATA_DIRECTORY_PROCESSED, 'horse_race_combined.csv'), index_col=0)
    race_combined['run_date'] = race_combined['run_date'].apply(lambda x: pd.Timestamp(x))
    race_combined = race_combined.sort_values(['horse_id', 'run_date'])
    race_combined.set_index(['horse_id', 'run_date'], inplace=True)

    if filter_first_occurrence:
        # Read data from those with first occurrence and drop some columns
        first_occurrence_df = pd.read_csv('{]{}'.format(DATA_DIRECTORY_PROCESSED, 'first_occurrence_race.csv'), index_col=0)
        first_occurrence_df['run_date'] = first_occurrence_df['run_date'].apply(pd.Timestamp)
        first_occurrence_df = first_occurrence_df.sort_values(['horse_id', 'run_date'])
        first_occurrence_df_index = first_occurrence_df.set_index(['horse_id', 'run_date']).index
    
        # Filter current dataframe by removing those without prior time information
        race_combined = race_combined[~race_combined.index.isin(first_occurrence_df_index)]

    if filter_columns:
        remove_dependent_columns(race_combined)
    return race_combined.reset_index() if reset_index else race_combined


def remove_dependent_columns(df):
    # Remove dependent-variable feature engineered columns not intended for training
    columns_to_drop = list(filter(lambda label: label != TRAINING_LABEL, DEPENDENT_COLUMNS))
    df.drop(columns_to_drop, axis=1, inplace=True)
    return columns_to_drop

def train_test_split_by_date(df):
    pass
