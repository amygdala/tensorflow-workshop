#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Convolutional Neural Network Estimator for MNIST, built with keras.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

import argparse
import os
import numpy as np
import time

import trainer.dataset as dataset

import tensorflow as tf

from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

FLAGS = None
BATCH_SIZE = 100

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


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["pixels"], [-1, 28, 28, 1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    K.set_learning_phase(1)
  else:
    K.set_learning_phase(0)

  conv1 = Convolution2D(32, (5, 5), activation='relu', input_shape=(28,28,1))(input_layer)
  pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
  conv2 = Convolution2D(64, (5, 5), activation='relu')(pool1)
  pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
  pool2_flat = Flatten()(pool2)
  dense = Dense(1024, activation='relu')(pool2_flat)
  dropout = Dropout(0.4)(dense)
  logits = Dense(10, activation='linear')(dropout)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),
     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
        export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output})

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  tf.summary.scalar('loss', loss)
  tf.summary.histogram('conv1', conv1)
  tf.summary.histogram('dense', dense)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
        return {'x': image_batch} , label_batch

    return _input_fn


def main(unused_argv):

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=FLAGS.job_dir)

  # Train and evaluate the model
  train_input = lambda: train_input_fn(
      FLAGS.data_dir,
      batch_size=BATCH_SIZE
  )
  eval_input = lambda: eval_input_fn(
      FLAGS.data_dir,
      batch_size=BATCH_SIZE
  )

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=2000)

  train_spec = tf.estimator.TrainSpec(train_input,
                                    max_steps=FLAGS.num_steps,
                                    hooks=[logging_hook]
                                    )
  def serving_input_receiver_fn():
      feature_tensor = tf.placeholder(tf.float32, [None, 784])
      return tf.estimator.export.ServingInputReceiver(
          {'pixels': feature_tensor}, {'pixels': feature_tensor})

  exporter = tf.estimator.FinalExporter('cnn_mnist', serving_input_receiver_fn)

  # While not shown here, we can also add a model 'exporter' to the EvalSpec.
  eval_spec = tf.estimator.EvalSpec(eval_input,
                                  steps=FLAGS.eval_steps,
                                  exporters=[exporter],
                                  name='cnn_mnist_keras'
                                  )

  tf.estimator.train_and_evaluate(mnist_classifier,
                                  train_spec,
                                  eval_spec)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/MNIST_data',
                      help='Directory for storing data')
  parser.add_argument('--job-dir', type=str,
                      default=os.path.join(
                          "/tmp/tfmodels/mnist_cnn_estimator",
                          str(int(time.time()))),
                      help='Directory for storing model info')
  parser.add_argument('--num_steps', type=int,
                      default=20000,
                      help='Number of training steps to run')
  parser.add_argument('--eval_steps', type=int,
                      default=100,
                      help='Number of eval steps to run')
  parser.add_argument('--logging_hook_iter', type=int,
                      default=5000,
                      help='How frequently to run the logging hook')
  FLAGS = parser.parse_args()
  tf.app.run()
