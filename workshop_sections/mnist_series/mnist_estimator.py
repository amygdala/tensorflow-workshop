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

"""Trains MNIST using tf.estimator.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import time

import tensorflow as tf
import dataset

ARGFLAGS = None

# comment out for less info during the training runs.
tf.logging.set_verbosity(tf.logging.INFO)


def train_input_fn(data_dir, batch_size=100):
  """Prepare data for training."""

  # When choosing shuffle buffer sizes, larger sizes result in better
  # randomness, while smaller sizes use less memory. MNIST is a small
  # enough dataset that we can easily shuffle the full epoch.
  ds = dataset.train(data_dir)
  ds = ds.cache().shuffle(buffer_size=50000).batch(batch_size=batch_size)

  # Iterate through the dataset a set number of times
  # during each training session.
  ds = ds.repeat(40)
  features = ds.make_one_shot_iterator().get_next()
  return {'pixels': features[0]}, features[1]


def eval_input_fn(data_dir, batch_size=100):
  features = dataset.test(data_dir).batch(
      batch_size=batch_size).make_one_shot_iterator().get_next()
  return {'pixels': features[0]}, features[1]


def define_and_run_linear_classifier(num_steps, logdir, batch_size=100):
    """Run a linear classifier."""
    feature_columns = [tf.contrib.layers.real_valued_column(
        "pixels", dimension=784)]

    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        model_dir=logdir
    )
    train_input = lambda: train_input_fn(
        ARGFLAGS.data_dir,
        batch_size=batch_size
    )
    linear_classifier.train(input_fn=train_input, steps=num_steps)

    print("Finished training.")

    # Evaluate accuracy.
    print("\n---Evaluating linear classifier accuracy...")
    eval_input = lambda: eval_input_fn(
        ARGFLAGS.data_dir,
        batch_size=batch_size
    )
    accuracy_score = linear_classifier.evaluate(
        input_fn=eval_input, steps=100)['accuracy']

    print('Linear Classifier Accuracy: {0:f}'.format(accuracy_score))


# After you've done a training run with optimizer learning rate 0.1,
# change it to 0.5 and run the training again.  Use TensorBoard to take
# a look at the difference.  You can see both runs by pointing it to the
# parent model directory, which by default is:
#
#   tensorboard --logdir=/tmp/tfmodels/mnist_estimators
def define_and_run_dnn_classifier(num_steps, logdir, lr=.1, batch_size=100):
    """Run a DNN classifier."""
    feature_columns = [tf.contrib.layers.real_valued_column("pixels", dimension=784)]

    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=[128, 32],
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=lr),
        model_dir=logdir
    )

    train_input = lambda: train_input_fn(
        ARGFLAGS.data_dir,
        batch_size=batch_size
    )
    eval_input = lambda: eval_input_fn(
        ARGFLAGS.data_dir,
        batch_size=batch_size
    )
    train_spec = tf.estimator.TrainSpec(train_input,
                                      max_steps=num_steps
                                      )

    # While not shown here, we can also add a model 'exporter' to the EvalSpec.
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                    steps=num_steps,
                                    name='fashion-eval'
                                    )
    tf.estimator.train_and_evaluate(dnn_classifier,
                                train_spec,
                                eval_spec)


def main(_):

    # Uncomment this if you'd like to run the linear classifier first.
    # print("\n-----Running linear classifier...")
    # model_dir = os.path.join(ARGFLAGS.model_dir, "linear_" + str(int(time.time())))
    # define_and_run_linear_classifier(ARGFLAGS.num_steps/2, model_dir, batch_size=ARGFLAGS.batch_size)

    print("\n---- Running DNN classifier...")
    model_dir = os.path.join(ARGFLAGS.model_dir, "deep_" + str(int(time.time())))
    define_and_run_dnn_classifier(ARGFLAGS.num_steps, model_dir, batch_size=ARGFLAGS.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/MNIST_data',
                        help='Directory for storing data')
    parser.add_argument('--model_dir', type=str,
                        default="/tmp/tfmodels/mnist_estimators",
                        help='Directory for storing model info')
    parser.add_argument('--num_steps', type=int,
                        default=15000,
                        help='Number of training steps to run')
    parser.add_argument('--batch_size', type=int,
                        default=100,
                        help='Batch size')
    ARGFLAGS = parser.parse_args()
    tf.app.run()