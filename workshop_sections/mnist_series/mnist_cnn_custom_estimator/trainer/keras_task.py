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
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

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

import trainer.mnist_input
from trainer.mnist_input import read_data_sets

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

FLAGS = None
BATCH_SIZE = 100

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

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


def generate_experiment_fn(**experiment_args):

  def _experiment_fn(run_config, hparams):

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": hparams.eval_data},
        y=hparams.eval_labels,
        num_epochs=1,
        shuffle=False)

    train_input = generate_input_fn(
        hparams.dataset,
        batch_size=BATCH_SIZE,
    )
    return tf.contrib.learn.Experiment(
    tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=FLAGS.job_dir, config=run_config),
        train_input_fn=train_input,
        eval_input_fn=eval_input_fn,
        **experiment_args
    )
  return _experiment_fn


def main(unused_argv):

  # Load training and eval data
  mnist = read_data_sets(FLAGS.data_dir,
      source_url=FLAGS.datasource_url)

  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  predict_data_batch = mnist.test.next_batch(20)

  def serving_input_receiver_fn():
      feature_tensor = tf.placeholder(tf.float32, [None, 784])
      return tf.estimator.export.ServingInputReceiver({'x': feature_tensor}, {'x': feature_tensor})

  learn_runner.run(
      generate_experiment_fn(
          min_eval_frequency=1,
          train_steps=FLAGS.num_steps,
          eval_steps=FLAGS.eval_steps,
          export_strategies=[saved_model_export_utils.make_export_strategy(
              serving_input_receiver_fn,
              exports_to_keep=1
          )]
      ),
      run_config = tf.contrib.learn.RunConfig().replace(model_dir=FLAGS.job_dir, save_checkpoints_steps=1000),
      hparams=hparam.HParams(dataset=mnist.train, eval_data=eval_data, eval_labels=eval_labels),
  )


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
  parser.add_argument('--datasource_url', type=str,
                      help='MNIST data source URL')
  FLAGS = parser.parse_args()
  tf.app.run()
