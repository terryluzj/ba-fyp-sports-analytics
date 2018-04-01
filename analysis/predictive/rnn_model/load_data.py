import pandas as pd
from analysis.predictive.feature_engineering import feature_engineer, drop_cols
from analysis.predictive.settings import DATA_DIRECTORY_FEATURE_ENGINEERED, DATA_DIRECTORY_PROCESSED, DEPENDENT_COLUMNS
from analysis.predictive.rnn_model.settings import TRAINING_LABEL, DATA_DIRECTORY, logger

DEPENDENT_COLUMNS_FORMATTED = list(map(lambda col_name: 'y_{}'.format(col_name), DEPENDENT_COLUMNS))
TRAINING_LABEL_FORMATTED = 'y_' + TRAINING_LABEL


def read_feature_engineered_dataframe(drop_rank_info=True, filter_columns=True,
                                      reuse=True, reuse_name=None, **params):
    # Read from feature engineered dataframe from file named reuse_name
    if reuse:
        assert reuse_name is not None
        logger.warning('Reusing stored feature engineered file %s.csv...' % reuse_name)
        race_combined_featured = pd.read_csv('{}{}.csv'.format(DATA_DIRECTORY_FEATURE_ENGINEERED, reuse_name),
                                             index_col=0)

        # Do type casting and multi-indexing
        race_combined_featured['run_date'] = race_combined_featured['run_date'].apply(lambda x: pd.Timestamp(x))
        race_combined_featured.set_index(['horse_id', 'run_date'], inplace=True)
    else:
        # Do feature engineering on the dataframe if specified (pass params for options defined in the function)
        logger.warning('Starting feature engineering on dataframe through read_race_dataframe')
        race_df = read_race_dataframe(filter_columns=filter_columns, reset_index=True, **params)
        if not filter_columns:
            # Rename dependent variable columns for recognition if not dropped later
            dependent_rename_dict = {before: 'y_' + before for before in DEPENDENT_COLUMNS}
            race_df.rename(columns=dependent_rename_dict, inplace=True)
        feature_race_df_name = 'rnn_featured' if reuse_name is None else reuse_name
        race_combined_featured_tuple = feature_engineer(race_df, df_name=feature_race_df_name)

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


def remove_dependent_columns(df, return_cols_dropped=False):
    # Remove dependent-variable feature engineered columns not intended for training
    columns_to_drop = list(filter(lambda label: label not in TRAINING_LABEL_FORMATTED, DEPENDENT_COLUMNS_FORMATTED))
    columns_dropped = []

    # Safely drop columns from the filtered list
    for col_name in columns_to_drop:
        try:
            # Add column name to the new list if successfully dropped
            df.drop(col_name, axis=1, inplace=True)
            columns_dropped.append(col_name)
        except ValueError:
            continue

    # Remove last run time information to match the definition of DV feature engineering
    if 'run_time_1000' not in TRAINING_LABEL:
        try:
            df.drop('last_run_time', axis=1, inplace=True)
            columns_dropped.append('last_run_time')
        except ValueError:
            pass

    # Return columns dropped if specified
    if return_cols_dropped:
        return columns_dropped

    # Return the dataframe after dropping columns
    return df


def train_validation_test_split_by_date(df_name, df=None, save=True,
                                        train_ratio=0.85, test_ratio=0.1, validation_ratio=0.05):
    # Function to split the dataset into three parts
    try:
        # Read data
        training_dataset = pd.read_csv('{}{}_training.csv'.format(DATA_DIRECTORY, df_name))
        testing_dataset = pd.read_csv('{}{}_testing.csv'.format(DATA_DIRECTORY, df_name))
        validation_dataset = pd.read_csv('{}{}_validation.csv'.format(DATA_DIRECTORY, df_name))
        logger.warning('Stored datasets found...')

        # Transform time value
        training_dataset['run_date'] = training_dataset['run_date'].apply(lambda x: pd.Timestamp(x))
        testing_dataset['run_date'] = testing_dataset['run_date'].apply(lambda x: pd.Timestamp(x))
        validation_dataset['run_date'] = validation_dataset['run_date'].apply(lambda x: pd.Timestamp(x))
    except FileNotFoundError:
        # Ensure a dataframe is used as a reference
        assert df is not None

        # Ensure that the ratio sums up to one
        if train_ratio != 0.8 or test_ratio != 0.1 or validation_ratio != 0.1:
            try:
                assert train_ratio + test_ratio + validation_ratio == 1
            except AssertionError:
                logger.warning('Split failed. Make sure ratios sum up to one...')

        # Ensure that time information is available
        if 'run_date' not in df.columns:
            # If run_date is already in the index level
            df_to_split = df.reset_index()
            try:
                # Raise Assertion Error if no date is found
                assert 'run_date' in df_to_split.columns
            except AssertionError:
                logger.warning('Split failed. No date information available for split...')
        else:
            # Get a copy for splitting to avoid cascaded value change
            df_to_split = df.copy()

        # Sort date value and get dataset index and date range
        df_to_split.sort_values('run_date', inplace=True)
        num_records = df_to_split.shape[0]
        training_index_end = int(num_records * train_ratio)
        testing_index_end = int(num_records * (train_ratio + test_ratio))
        training_last_date = df_to_split.iloc[training_index_end]['run_date']
        testing_last_date = df_to_split.iloc[testing_index_end]['run_date']

        # Split dataset accordingly
        training_dataset = df_to_split[df_to_split['run_date'] < training_last_date]
        next_date = (df_to_split['run_date'] >= training_last_date) & (df_to_split['run_date'] < testing_last_date)
        testing_dataset = df_to_split[next_date]
        validation_dataset = df_to_split[df_to_split['run_date'] >= testing_last_date]

        # Store the dataset if specified
        if save:
            training_dataset.to_csv('{}{}_training.csv'.format(DATA_DIRECTORY, df_name))
            testing_dataset.to_csv('{}{}_testing.csv'.format(DATA_DIRECTORY, df_name))
            validation_dataset.to_csv('{}{}_validation.csv'.format(DATA_DIRECTORY, df_name))

    # Report date range
    logger.warning('Training: {} -> {}'.format(training_dataset['run_date'].min(), training_dataset['run_date'].max()))
    logger.warning('Testing: {} -> {}'.format(testing_dataset['run_date'].min(), testing_dataset['run_date'].max()))
    logger.warning('Validation: {} -> {}'.format(validation_dataset['run_date'].min(),
                                                 validation_dataset['run_date'].max()))

    # Set multi-index and return
    training_dataset = training_dataset.set_index(['horse_id', 'run_date']).astype(float)
    testing_dataset = testing_dataset.set_index(['horse_id', 'run_date']).astype(float)
    validation_dataset = validation_dataset.set_index(['horse_id', 'run_date']).astype(float)
    return training_dataset, testing_dataset, validation_dataset


if __name__ == '__main__':
    # Testing statements
    # race_df = read_race_dataframe(include_first_occurrence=True, reset_index=False, filter_columns=False)
    feature_df = read_feature_engineered_dataframe(reuse=True, reuse_name='rnn_featured',
                                                   drop_rank_info=True, include_first_occurrence=True,
                                                   filter_columns=False)
    test_df = train_validation_test_split_by_date('race_record_first_included', df=feature_df)
    pass
