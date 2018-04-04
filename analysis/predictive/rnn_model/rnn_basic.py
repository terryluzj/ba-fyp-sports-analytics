import numpy as np
import tensorflow as tf

from analysis.predictive.model_comparer import ModelComparer
from analysis.predictive.rnn_model.pipeline import get_train_test_set
from analysis.predictive.rnn_model.settings import FILE_DIRECTORY, TRAINING_LABEL, PREDICT_FIRST_RACE
from analysis.predictive.rnn_model.settings import logger, get_current_training_process
from datetime import datetime

# CONFIG =============================================================================================

TIME_STEP = 15
CONFIG = {
    # Dataset related
    'file_name': 'race_record_first_included',
    'target_column': TRAINING_LABEL,
    'first_race_record': PREDICT_FIRST_RACE,

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
    'test_batch_size': 500,

    # Logging related
    'print_interval': 10
}
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'log'
logdir = '{}/run-{}-{}-first-{}/'.format(root_logdir, now, TRAINING_LABEL, CONFIG['first_race_record'])

# DATA PREPARATION =============================================================================================

train_test_set = get_train_test_set(target_column=CONFIG['target_column'], max_length=CONFIG['max_length'],
                                    file_name=CONFIG['file_name'], first_race_record=CONFIG['first_race_record'])
train_X, train_y, train_mapped, train_seq_length = train_test_set['train']
test_X, test_y, test_mapped, test_seq_length = train_test_set['test']

# CONSTRUCTION PHASE ===========================================================================================

# Get configuration of model
n_steps = CONFIG['n_steps']
n_inputs = CONFIG['n_inputs']
n_neurons = CONFIG['n_neurons']
n_outputs = CONFIG['n_outputs']
learning_rate = CONFIG['learning_rate']
operator = ModelComparer.get_operator(CONFIG['target_column'])

# Set placeholder, cell and outputs
logger.warning('Constructing graph...')
X = tf.placeholder(tf.float64, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float64, [None, n_steps, n_outputs])
y_map = tf.placeholder(tf.float64, [None, n_steps, n_outputs])
sequence_length = tf.placeholder(tf.int32, [None])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
                                              output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, sequence_length=sequence_length)

# Set loss function and optimizer
if operator == 'diff':
    loss = tf.reduce_mean(tf.square(tf.add(outputs, y_map) - tf.add(y, y_map)))
elif operator == 'quo':
    loss = tf.reduce_mean(tf.square(tf.multiply(outputs, y_map) - tf.multiply(y, y_map)))
else:
    loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Initialize all graph nodes
init = tf.global_variables_initializer()
saver = tf.train.Saver()
mse_summary = tf.summary.scalar('RMSE', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# EXECUTION PHASE ==============================================================================================

# Get configuration of execution
num_instances = train_X.shape[0]
n_epochs = CONFIG['n_epochs']
batch_size = CONFIG['batch_size']
test_batch_size = CONFIG['test_batch_size']
n_iteration = num_instances // batch_size

with tf.Session() as sess:
    # Run the training session

    init.run()
    for epoch in range(n_epochs):

        # Set up start and end index of batch set
        start_idx = 0
        end_idx = start_idx + batch_size

        for iteration in range(n_iteration):

            # Get X, y and sequence length batch
            X_batch = train_X[start_idx:end_idx]
            y_batch = train_y[start_idx:end_idx]
            seq_length_batch = train_seq_length[start_idx:end_idx]

            # Get test batch randomly
            start_idx_test = np.random.randint(0, test_X.shape[0] - test_batch_size)
            end_idx_test = start_idx_test + test_batch_size
            X_batch_test = test_X[start_idx_test:end_idx_test]
            y_batch_test = test_y[start_idx_test:end_idx_test]
            seq_length_batch_test = test_seq_length[start_idx_test:end_idx_test]

            # Run training on the batched set
            feed_dict = {X: X_batch, y: y_batch, sequence_length: seq_length_batch}
            feed_dict_test = {X: X_batch_test, y: y_batch_test, sequence_length: seq_length_batch_test}
            try:
                y_map_batch = train_mapped[start_idx:end_idx]
                y_map_batch_test = test_mapped[start_idx_test:end_idx_test]
                feed_dict[y_map] = y_map_batch
                feed_dict_test[y_map] = y_map_batch_test
            except TypeError:
                pass
            sess.run(training_op, feed_dict=feed_dict)

            # Calculate RMSE
            train_mse = loss.eval(feed_dict=feed_dict)
            test_mse = loss.eval(feed_dict=feed_dict_test)
            train_rmse = train_mse ** (1/2)
            test_mse = test_mse ** (1/2)

            # Log information
            if iteration % CONFIG['print_interval'] == 0:
                # Write to Tensorboard
                summary_str = mse_summary.eval(feed_dict=feed_dict)
                step = epoch * n_iteration + iteration
                file_writer.add_summary(summary_str, step)

                # Print out loss information
                percentage = int(iteration / n_iteration * 100)
                training_err_str = 'Training error: %4.3f' % train_mse
                testing_err_str = 'Testing error: %4.3f' % test_mse
                message = '[Epoch %d] %s (%s - %s)'
                message = message % (epoch + 1, get_current_training_process(percentage),
                                     training_err_str, testing_err_str)
                logger.warning(message)

            # Update start and end index
            start_idx = end_idx
            end_idx = start_idx + batch_size

        # Save model
        save_path = saver.save(sess, FILE_DIRECTORY + 'model/basic_rnn_%s.ckpt' % TRAINING_LABEL)

    save_path = saver.save(sess, FILE_DIRECTORY + 'model/basic_rnn_final_%s.ckpt' % TRAINING_LABEL)
