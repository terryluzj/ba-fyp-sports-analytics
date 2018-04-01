import numpy as np

from analysis.predictive.rnn_model.load_data import train_validation_test_split_by_date
from analysis.predictive.rnn_model.settings import logger


def load_data(file_name='race_record_first_included'):
    # Load training and testing data from the function defined in load_data.py
    df_name = file_name if '.csv' not in file_name else file_name.replace('.csv', '')

    # Return training, testing and validation split of the dataset
    train, test, validation = train_validation_test_split_by_date(df_name=df_name)
    return train, test, validation


def transform_dataset(df, target_column='y_run_time_1000'):
    # Transform the dataset into X and y, with some DV feature engineering
    feature_names = list(filter(lambda col_name: 'y_run' in col_name, df.columns))

    # Initiate mapping for DV feature engineering
    mapped = None

    # Drop the last run_time information if target column is not run_time_1000
    if target_column != 'y_run_time_1000':
        feature_names.append('last_run_time')

        # Check the target column name and do feature engineering accordingly
        quo_feature = '_quo' in target_column
        diff_feature = '_diff' in target_column
        quo_or_diff = quo_feature or diff_feature

        if quo_or_diff:
            if target_column in df.columns:
                # Remove zeroes
                df_transformed = df.loc[df[target_column] != 0].copy()

                # Reserve the target column if found in the column list
                feature_names.remove(target_column)
                mapped = df_transformed['y_run_time_1000']
            else:
                # Get the original target column
                original_target_name = target_column.replace('_quo' if quo_feature else '_diff', '')

                # Remove zeroes
                df_transformed = df.loc[df[original_target_name] != 0].copy()

                # Assign it to the mapping and do the calculation as new series
                original_target = df[original_target_name]
                mapped = original_target
                if quo_feature:
                    df_transformed[target_column] = df_transformed['y_run_time_1000'] / \
                                                    df_transformed[original_target_name]
                else:
                    df_transformed[target_column] = df_transformed['y_run_time_1000'] - \
                                                    df_transformed[original_target_name]
        else:
            # Log error information
            logger.warning('No dataframe returned as target_column was not specified correctly')
            return {}
    else:
        # Remove zeroes
        df_transformed = df.loc[df['last_run_time'] != 0].copy()

        # Do not remove the dependent column if matched with the default target column
        feature_names.remove('y_run_time_1000')

    # Drop all irrelevant DV features
    df_transformed.drop(feature_names, axis=1, inplace=True)

    # Split into X and y and return the final dataframe
    x_df = df_transformed.drop(target_column, axis=1)
    y_df = df_transformed[target_column]
    return {'X': x_df, 'y': y_df, 'mapped': mapped}


def get_matrix_combination(df, max_length=None, drop_col_name='run_date', groupby_col_name='horse_id'):
    # Function to convert dataframe to arrays in sequential format
    # Race record of one horse will be compress into one single two-dimensional array
    # Final returned matrix will be in shape (record_length, max_length, feature_length)

    def get_padded_array(arr_lst, length_max):
        # Do padding to meet the sequential format
        padding = max_length - len(arr_lst)
        if padding > 0:
            # Add zeros array to the end of the original one
            for idx in range(length_max - len(arr_lst)):
                arr_lst.append(np.zeros(arr_lst[idx].shape[0]))
            return arr_lst
        else:
            # Remove those aligned to the left if the total length exceeds the maximum
            return arr_lst[-padding:]

    if drop_col_name is not None:
        # Drop column safely
        df_dropped = df.reset_index().drop(drop_col_name, axis=1)
    else:
        df_dropped = df.copy()

    # Group by a column name and convert back to array form
    df_matrix = df_dropped.groupby(groupby_col_name).apply(lambda x: x.as_matrix())
    df_matrix = df_matrix.apply(lambda arr_lst: list(map(lambda arr: arr[1:], arr_lst)))
    if max_length is None:
        # Get maximum length of current dataset if set to be None
        max_length = df_matrix.apply(lambda arr_lst: len(arr_lst)).max()

    # Do padding and type casting
    length = df_matrix.apply(lambda arr_lst: min(max_length, len(arr_lst)))
    df_matrix = df_matrix.apply(lambda arr_lst: np.array(get_padded_array(arr_lst, max_length)))
    return {'transformed': np.array(list(df_matrix)), 'length': np.array(list(length))}


""" Example code from Tensorflow documentation
def train_input_fn(features, labels, batch_size, count=None):
    # Input function for training, example code from Iris classification example on Tensorflow
    assert batch_size is not None

    # Convert the input Pandas dataframe to a Tensorflow Dataset object and do the shuffling
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(len(features)).repeat(count=count).batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    # Input function for training, example code from Iris classification example on Tensorflow
    assert batch_size is not None and labels is not None

    # Transform features and labels as Tensorflow Dataset and batch the data
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset
"""
