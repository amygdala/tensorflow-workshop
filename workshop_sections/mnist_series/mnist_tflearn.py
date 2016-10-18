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

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ARGFLAGS = None

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


def run_linear_classifier():
    """Run a linear classifier."""
    # read in data, downloading first if necessary
    print("Downloading and reading data sets...")
    data_sets = input_data.read_data_sets(ARGFLAGS.data_dir)
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        data_sets.train.images)
    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns, n_classes=10)
    classifier.fit(data_sets.train.images,
                   data_sets.train.labels.astype(numpy.int64),
                   batch_size=100, steps=10000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        data_sets.test.images,
        data_sets.test.labels.astype(numpy.int64))['accuracy']
    print('Linear Classifier Accuracy: {0:f}'.format(accuracy_score))


def run_dnn_classifier():
    """Run a DNN classifier."""
    # read in data, downloading first if necessary
    data_sets = input_data.read_data_sets(ARGFLAGS.data_dir)
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        data_sets.train.images)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, n_classes=10,
        hidden_units=[128, 32],
        # After you've done a training run with optimizer learning rate 0.1,
        # change it to 0.5 and run the training again.  Use TensorBoard to take
        # a look at the difference.  You can see both runs by pointing it to the
        # parent model directory, which by default is:
        #   tensorboard --logdir=/tmp/tfmodels/mnist_tflearn
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        model_dir=ARGFLAGS.model_dir
        )
    classifier.fit(data_sets.train.images,
                   data_sets.train.labels.astype(numpy.int64),
                   batch_size=100, max_steps=ARGFLAGS.num_steps)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        data_sets.test.images,
        data_sets.test.labels.astype(numpy.int64))['accuracy']
    print('DNN Classifier Accuracy: {0:f}'.format(accuracy_score))


def main(_):

    # print("\n-----Running linear classifier...")
    # run_linear_classifier()
    print("\n---- Running DNN classifier...")
    run_dnn_classifier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST_data',
                        help='Directory for storing data')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(
                            "/tmp/tfmodels/mnist_tflearn",
                            str(int(time.time()))),
                        help='Directory for storing model info')
    parser.add_argument('--num_steps', type=int,
                        default=25000,
                        help='Number of training steps to run')
    ARGFLAGS = parser.parse_args()
    tf.app.run()
