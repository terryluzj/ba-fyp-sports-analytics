import tensorflow as tf

from analysis.predictive.rnn_model.load_data import train_validation_test_split_by_date


def load_data(file_name='race_record_first_included'):
    # Load training and testing data from the function defined in load_data.py
    df_name = file_name if '.csv' not in file_name else file_name.replace('.csv', '')

    # Return training, testing and validation split of the dataset
    train, test, validation = train_validation_test_split_by_date(df_name=df_name)
    return train, test, validation


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
