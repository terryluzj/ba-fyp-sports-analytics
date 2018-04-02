import numpy as np
import tensorflow as tf

from analysis.predictive.rnn_model.pipeline import get_train_test_set
from analysis.predictive.rnn_model.settings import TRAINING_LABEL, logger, get_current_training_process

TIME_STEP = 15
CONFIG = {
    # Dataset related
    'file_name': 'race_record_first_included',
    'target_column': TRAINING_LABEL,
    
    # Construction phase related
    'max_length': TIME_STEP,
    'n_steps': TIME_STEP,
    'n_inputs': 215 if TRAINING_LABEL == 'y_run_time_1000' else 214,
    'n_neurons': 150,
    'n_outputs': 1,
    'learning_rate': 0.001,

    # Execution phase related
    'n_epochs': 50,
    'batch_size': 100,
    'testing_data_size': 1000,

    # Logging related
    'print_interval': 20
}

# DATA PREPARATION =============================================================================================

train_test_set = get_train_test_set(target_column=CONFIG['target_column'], max_length=CONFIG['max_length'],
                                    file_name=CONFIG['file_name'])
train_X, train_y, train_mapped, train_seq_length = train_test_set['train']
test_X, test_y, test_mapped, test_seq_length = train_test_set['test']

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
            start_idx_test = np.random.randint(0, test_X.shape[0] - batch_size)
            end_idx_test = start_idx_test + batch_size
            X_batch_test = test_X[start_idx_test:end_idx_test]
            y_batch_test = test_y[start_idx_test:end_idx_test]
            seq_length_batch_test = test_seq_length[start_idx_test:end_idx_test]

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
            if iteration % CONFIG['print_interval'] == 0:
                percentage = int(iteration / n_iteration * 100)
                training_err_str = 'Training error: %4.3f' % train_mse
                testing_err_str = 'Testing error: %4.3f' % test_mse
                message = '[Epoch %d] %s (%s - %s)'
                message = message % (epoch, get_current_training_process(percentage), training_err_str, testing_err_str)
                logger.warning(message)
