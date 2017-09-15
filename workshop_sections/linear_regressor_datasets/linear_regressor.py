# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Linear regression using the LinearRegressor Estimator and Datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import time

import tensorflow as tf

import datasets_input

FLAGS = None
PRICE_NORM_FACTOR = 1000
# with this simple example, we'll use these two fields for training.
TRAIN_FEATURE_NAMES = ['curb-weight','highway-mpg']



def main(argv):
  """Builds, trains, and evaluates the model."""

  # Get the input Datasets
  (train, test) = datasets_input.dataset()

  # Apply a map() to further transform the data - switch the labels to units
  # of thousands for better convergence.
  def to_thousands(features, labels):
    return features, labels/PRICE_NORM_FACTOR
  train = train.map(to_thousands)
  test = test.map(to_thousands)

  # Build the training input_fn.
  def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  # Build the validation input_fn.
  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  # All the columns we want to use are numeric, so we can construct
  # the feature column info like this. In general, different columns might have
  # different types.
  feature_columns = [tf.feature_column.numeric_column(key=i) for i in TRAIN_FEATURE_NAMES]
  # Build the Estimator.
  model = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                       model_dir=FLAGS.model_dir)

  # Train the model.
  # By default, the Estimators log output every 100 steps.
  model.train(input_fn=input_train, steps=FLAGS.num_steps)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=input_test)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]
  print("Average loss: %s" % average_loss)

  # input data element items correspond to TRAIN_FEATURE_NAMES list, i.e.:
  # [curb-weight, highway-mpg]
  prediction_input = [[2000, 30], [3000, 40]]

  def predict_input_fn():
    def decode(x):
        x = tf.split(x, 2) # Need to split into our 2 features
        # When predicting, we don't need (or have) any labels
        return dict(zip(TRAIN_FEATURE_NAMES, x)) # Then build a dict from them
    # The from_tensor_slices function will use a memory structure as input
    dataset = tf.contrib.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None # In prediction, we have no labels

  predict_results = model.predict(input_fn=predict_input_fn)

  # Print the prediction results.
  print("\nPrediction results:")
  for i, prediction in enumerate(predict_results):
    print("i %s, prediction %s" % (i, prediction))
    msg = ("Curb weight: {: 4d}lbs, "
           "Highway: {: 0d}mpg, "
           "Prediction: ${: 9.2f}")
    msg = msg.format(prediction_input[i][0], prediction_input[i][1],
                     PRICE_NORM_FACTOR*prediction["predictions"][0])
    print("    " + msg)
  print()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', type=str,
                      default=os.path.join(
                          "/tmp/tfmodels/linear_regressor",
                          str(int(time.time()))),
                      help='Directory for storing model info')
  parser.add_argument('--num_steps', type=int,
                      default=1000,
                      help='Number of training steps to run')
  FLAGS = parser.parse_args()
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
