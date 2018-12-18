import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
EVAL_INTERVAL_SECS = 5

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model_save_dir/'


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        y_ = mnist_inference.inference(x)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH).\
            model_checkpoint_path

        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            global_step = ckpt.split('/')[-1].split('-')[-1]
            validate_feed = {x: mnist.validation.images,
                             y: mnist.validation.labels}
            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            # print('step %s: validation accuracy is %g.' % (global_step, accuracy_score))
    # print('step %s: validation accuracy is %g.' % (global_step, accuracy_score))
    return accuracy_score


