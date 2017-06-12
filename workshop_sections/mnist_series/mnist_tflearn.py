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

"""Trains MNIST using tf.contrib.learn.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ARGFLAGS = None
DATA_SETS = None
BATCH_SIZE = 40

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


# example calls: generate_input_fn(DATA_SETS.train) or generate_input_fn(DATA_SETS.test)
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
        return {'pixels': image_batch} , label_batch

    return _input_fn


def define_and_run_linear_classifier(num_steps, logdir, batch_size=BATCH_SIZE):
    """Run a linear classifier."""
    feature_columns = [tf.contrib.layers.real_valued_column(
        "pixels", dimension=784)]

    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        model_dir=logdir
    )
    classifier.fit(
        input_fn=generate_input_fn(DATA_SETS.train, batch_size=batch_size),
        steps=num_steps)

    print("Finished training.")

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=generate_input_fn(DATA_SETS.test, batch_size), steps=100)['accuracy']

    print('Linear Classifier Accuracy: {0:f}'.format(accuracy_score))


def define_and_run_dnn_classifier(num_steps, logdir, lr=.1, batch_size=40):
    """Run a DNN classifier."""
    feature_columns = [tf.contrib.layers.real_valued_column("pixels", dimension=784)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=[128, 32],
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=lr),
        model_dir=logdir
    )
    # After you've done a training run with optimizer learning rate 0.1,
        # change it to 0.5 and run the training again.  Use TensorBoard to take
        # a look at the difference.  You can see both runs by pointing it to the
        # parent model directory, which by default is:
        #
        #   tensorboard --logdir=/tmp/tfmodels/mnist_tflearn

    classifier.fit(input_fn=generate_input_fn(DATA_SETS.train, batch_size=batch_size),
                   steps=num_steps)

    print("Finished running the deep training via the fit() method")

    print("\n---Evaluating DNN classifier accuracy...")
    accuracy_score = classifier.evaluate(input_fn=generate_input_fn(DATA_SETS.test, batch_size=batch_size),
                                         steps=100)['accuracy']

    print('DNN Classifier Accuracy: {0:f}'.format(accuracy_score))


def main(_):

    # read in data, downloading first if necessary
    global DATA_SETS
    print("Downloading and reading data sets...")
    DATA_SETS = input_data.read_data_sets(ARGFLAGS.data_dir)

    # Uncomment this if you'd like to run the linear classifier first.
    # print("\n-----Running linear classifier...")
    # model_dir = os.path.join("/tmp/tfmodels/mnist_tflearn", str(int(time.time())))
    # define_and_run_linear_classifier(ARGFLAGS.num_steps, model_dir)

    print("\n---- Running DNN classifier...")
    define_and_run_dnn_classifier(ARGFLAGS.num_steps, ARGFLAGS.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/MNIST_data',
                        help='Directory for storing data')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(
                            "/tmp/tfmodels/mnist_tflearn",
                            str(int(time.time()))),
                        help='Directory for storing model info')
    parser.add_argument('--num_steps', type=int,
                        default=15000,
                        help='Number of training steps to run')
    ARGFLAGS = parser.parse_args()
    tf.app.run()
