#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains MNIST using a custom estimator, with the model based on the one here:
https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts ,
and using tf.contrib.layers to build the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import ModeKeys
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
DATA_SETS = None
BATCH_SIZE = 40

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)

def generate_input_fn(dataset, batch_size=BATCH_SIZE):
    def _input_fn():
        X = tf.constant(dataset.images)
        Y = tf.constant(dataset.labels, dtype=tf.int32)
        image_batch, label_batch = tf.train.shuffle_batch([X,Y],
                               batch_size=batch_size,
                               capacity=8*batch_size,
                               min_after_dequeue=4*batch_size,
                               enqueue_many=True
                              )
        return {'inputs': image_batch} , label_batch

    return _input_fn


def model_fn(features, labels, mode, params):
    """Model function for Estimator."""

    # labels = tf.Print(labels, [labels], message="labels is: ")

    y_ = tf.cast(labels, tf.float32)  # is this still necessary?
    # hmmm
    # y_ = tf.reshape(y_, [-1])
    x = features.get('inputs')
    # x = tf.Print(x, [x], message="x is: ")


    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    h_conv1 = layers.convolution2d(x_image, 32, [5,5])
    h_pool1 = layers.max_pool2d(h_conv1, [2,2])

    # second convolutional layer
    h_conv2 = layers.convolution2d(h_pool1, 64, [5,5])
    h_pool2 = layers.max_pool2d(h_conv2, [2,2])

    # densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = layers.fully_connected(h_pool2_flat, 1024)
    h_fc1_drop = layers.dropout(
        h_fc1, keep_prob=params["dropout"],
        is_training=(mode == ModeKeys.TRAIN))

    # readout layer
    y_conv = layers.fully_connected(h_fc1_drop, 10, activation_fn=None)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    global_step = tf.train.get_global_step()
    train_op = tf.contrib.layers.optimize_loss(  # change this?
        loss=cross_entropy,
        global_step=global_step,
        learning_rate=params["learning_rate"],
        optimizer="Adam")

    predictions = tf.argmax(y_conv, 1)
    prediction_output = tf.estimator.export.PredictOutput({'accuracy': predictions})
    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=cross_entropy,
            train_op=train_op,
            export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output}
        )
    # return predictions, cross_entropy, train_op


def run_cnn_classifier():
    """Run a CNN classifier using a custom Estimator."""

    # print("Downloading and reading data sets...")
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Set model params
    model_params = {"learning_rate": 1e-4, "dropout": 0.5}

    cnn = tf.estimator.Estimator(
        model_fn=model_fn, params=model_params,
        model_dir=FLAGS.model_dir)

    print("Starting training for %s steps max" % FLAGS.num_steps)
    cnn.train(generate_input_fn(DATA_SETS.train, batch_size=BATCH_SIZE), steps=FLAGS.num_steps)
    # cnn.train(x=mnist.train.images,
    #         y=mnist.train.labels, batch_size=50,
    #         max_steps=FLAGS.num_steps)

    # Evaluate accuracy.
    # Evaluate accuracy.
    result = cnn.evaluate(
        input_fn=generate_input_fn(DATA_SETS.test, batch_size=BATCH_SIZE), steps=100)['accuracy']
    print("eval result: %s" % result)
    # accuracy_score = cnn.evaluate(
        # input_fn=generate_input_fn(DATA_SETS.test, batch_size=BATCH_SIZE), steps=100)['accuracy']
    # print('DNN Classifier Accuracy: {0:f}'.format(accuracy_score))

    # print(cnn.evaluate(DATA_SETS.test.images, DATA_SETS.test.labels))

    # Print out some predictions, just drawn from the test data.
    # batch = DATA_SETS.test.next_batch(20)
    # predictions = cnn.predict(x=batch[0], as_iterable=True)
    # for i, p in enumerate(predictions):
    #     print("Prediction: %s for correct answer %s" %
    #           (p, list(batch[1][i]).index(1)))


def main(_):

    # read in data, downloading first if necessary
    global DATA_SETS
    print("Downloading and reading data sets...")
    DATA_SETS = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    run_cnn_classifier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/MNIST_data',
                        help='Directory for storing data')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(
                            "/tmp/tfmodels/mnist_cnn_estimator",
                            str(int(time.time()))),
                        help='Directory for storing model info')
    parser.add_argument('--num_steps', type=int,
                        default=5000,
                        help='Number of training steps to run')
    FLAGS = parser.parse_args()
    tf.app.run()
