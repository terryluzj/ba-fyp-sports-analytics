import numpy as np
import tensorflow as tf

from analysis.predictive.rnn_model.pipeline import load_data, transform_dataset, get_matrix_combination
from analysis.predictive.rnn_model.settings import TRAINING_LABEL, logger, get_current_training_process

CONFIG = {
    # Dataset related
    'file_name': 'race_record_first_included',
    'target_column': TRAINING_LABEL,
    
    # Construction phase related
    'max_length': 20,
    'n_steps': 20,
    'n_inputs': 215 if TRAINING_LABEL == 'y_run_time_1000' else 214,
    'n_neurons': 100,
    'n_outputs': 1,
    'learning_rate': 0.001,

    # Execution phase related
    'n_epochs': 50,
    'batch_size': 100,
    'testing_data_size': 1000,
}

# DATA PREPARATION =============================================================================================

# Get configuration
max_length = CONFIG['max_length']

# Get training and testing set
logger.warning('Current training label is {}. Fetching training and testing set...'.format(TRAINING_LABEL))
train, test, validation = load_data(file_name=CONFIG['file_name'])

# Get X, y, mapping series of training and testing set
logger.warning('Transforming training and testing set...')
train_transformed = transform_dataset(train, target_column=CONFIG['target_column'])
test_transformed = transform_dataset(test, target_column=CONFIG['target_column'])

# Get matrix transformation
logger.warning('Getting matrix representation of training and testing set...')
for key in train_transformed.keys():
    curr_series = train_transformed[key]
    if curr_series is not None:
        train_transformed[key] = get_matrix_combination(curr_series, max_length=max_length)
for key in test_transformed.keys():
    curr_series = test_transformed[key]
    if curr_series is not None:
        test_transformed[key] = get_matrix_combination(curr_series, max_length=max_length)

# Assign the variables
train_X = train_transformed['X']['transformed']
train_y = train_transformed['y']['transformed']
train_mapped = train_transformed['mapped']
if train_mapped is not None:
    train_mapped = train_transformed['mapped']['transformed']
train_seq_length = train_transformed['X']['length']

testing_data_size = CONFIG['testing_data_size']
test_X = test_transformed['X']['transformed']
test_y = test_transformed['y']['transformed']
test_mapped = test_transformed['mapped']
if test_mapped is not None:
    test_mapped = test_transformed['mapped']['transformed']
test_seq_length = test_transformed['X']['length']

# CONSTRUCTION PHASE ===========================================================================================

# Get configuration of model
n_steps = CONFIG['n_steps']
n_inputs = CONFIG['n_inputs']
n_neurons = CONFIG['n_neurons']
n_outputs = CONFIG['n_outputs']
learning_rate = CONFIG['learning_rate']

# Set placeholder, cell and outputs
logger.warning('Constructing graph...')
X = tf.placeholder(tf.float64, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float64, [None, n_steps, n_outputs])
sequence_length = tf.placeholder(tf.int32, [None])
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
                                              output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, sequence_length=sequence_length)

# Set loss function and optimizer
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Initialize all graph nodes
init = tf.global_variables_initializer()

# EXECUTION PHASE ==============================================================================================

# Get configuration of execution
num_instances = train_X.shape[0]
n_epochs = CONFIG['n_epochs']
batch_size = CONFIG['batch_size']
n_iteration = num_instances // batch_size

with tf.Session() as sess:
    # Run the training session

    init.run()
    for epoch in range(n_epochs):

        # Set up start and end index of batch set
        start_idx = 0
        end_idx = start_idx + batch_size

        for iteration in range(n_iteration):

            # Get test batch randomly
            # TODO: Reconstruct sampling technique for testing set
            test_start_idx = np.random.randint(0, test_X.shape[0] - batch_size)
            test_end_idx = test_start_idx + batch_size
            X_batch_test = test_X[test_start_idx:test_end_idx]
            y_batch_test = test_y[test_start_idx:test_end_idx]
            seq_length_batch_test = test_seq_length[test_start_idx:test_end_idx]

            # Get X, y and sequence length batch
            X_batch = train_X[start_idx:end_idx]
            y_batch = train_y[start_idx:end_idx]
            seq_length_batch = train_seq_length[start_idx:end_idx]

            # Update start and end index
            start_idx = end_idx
            end_idx = start_idx + batch_size

            # Run training on the batched set
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, sequence_length: seq_length_batch})

            # Calculate RMSE
            train_mse = loss.eval(feed_dict={X: X_batch, y: y_batch, sequence_length: seq_length_batch})
            test_mse = loss.eval(feed_dict={X: X_batch_test, y: y_batch_test, sequence_length: seq_length_batch_test})
            train_rmse = train_mse ** (1/2)
            test_mse = test_mse ** (1/2)

            # Log information
            if iteration % 10 == 0:
                percentage = int(iteration / n_iteration * 100)
                training_err_str = 'Training error: %4.3f' % train_mse
                testing_err_str = 'Testing error: %4.3f' % test_mse
                message = '[Epoch %d] %s (%s - %s)'
                message = message % (epoch, get_current_training_process(percentage), training_err_str, testing_err_str)
                logger.warning(message)
