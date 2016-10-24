# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math

import tensorflow as tf
from tensorflow.contrib.learn import (
    ModeKeys, Experiment, Estimator, learn_runner)

import input_utils


def make_model_fn(index,
                  num_partitions=1,
                  embedding_size=128,
                  vocab_size=2 ** 15,
                  num_sim=8,
                  num_sampled=64,
                  learning_rate=0.1):
  def _make_model(target_words, context_words, mode):
    index_tensor = tf.constant(index)
    reverse_index = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            index_tensor, tf.constant(range(1, vocab_size), dtype=tf.int64)
        ),
        0
    )

    # tf.contrib.learn.Estimator.fit adds an addition dimension to input
    target_words_squeezed = tf.squeeze(target_words, squeeze_dims=[1])
    target_indices = reverse_index.lookup(target_words_squeezed)

    with tf.device(tf.train.replica_device_setter()):
      with tf.variable_scope('nce',
                             partitioner=tf.fixed_size_partitioner(
                                 num_partitions)):

        embeddings = tf.get_variable(
            'embeddings',
            shape=[vocab_size, embedding_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-1.0, 1.0)
        )
        if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
          nce_weights = tf.get_variable(
              'nce_weights',
              shape=[vocab_size, embedding_size],
              dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(
                  stddev=1.0 / math.sqrt(embedding_size)
              )
          )
          nce_biases = tf.get_variable(
              'nce_biases',
              initializer=tf.zeros_initializer([vocab_size]),
              dtype=tf.float32
          )

      prediction_dict, loss, train_op = ({}, None, None)

      if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
        context_indices = tf.expand_dims(
            reverse_index.lookup(context_words), 1)
        embedded = tf.nn.embedding_lookup(embeddings, target_indices)

        loss = tf.reduce_mean(tf.nn.nce_loss(
            nce_weights,
            nce_biases,
            embedded,
            context_indices,
            num_sampled,
            vocab_size
        ))
        tf.scalar_summary('loss', loss)

      if mode == ModeKeys.TRAIN:
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=tf.contrib.framework.get_global_step()
        )

      if mode in [ModeKeys.EVAL, ModeKeys.INFER]:
        # Compute the cosine similarity between examples and embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, target_indices)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
        prediction_dict['values'], predictions = tf.nn.top_k(
            similarity, sorted=True, k=num_sim)
        index_tensor = tf.concat(0, [tf.constant(['UNK']), index_tensor])
        prediction_dict['predictions'] = tf.gather(index_tensor, predictions)

      return prediction_dict, loss, train_op
  return _make_model


def make_experiment_fn(args):
  index = input_utils.vector_from_gcs(args.index_file)
  train_input_fn = input_utils.from_tfrecords(
      args.train_data_paths,
      args.batch_size,
      num_epochs=args.num_epochs
  )
  eval_input_fn = input_utils.eval_input(args.eval_words)

  def experiment_fn(output_dir):
    return Experiment(
        Estimator(
            model_fn=make_model_fn(index, **args.hyper_params),
            model_dir=output_dir
        ),
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )
  return experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--index-file',
      help="""\
      A .npy file in GCS which contains a single string vector,
      the words in the vocabulary, ordered by frequency
      """,
      required=True
  )
  parser.add_argument(
      '--train-data-paths',
      help='TFRecord files for training',
      required=True
  )
  parser.add_argument(
      '--output-path',
      help='GCS path to output files',
  )
  parser.add_argument(
      '--eval-words',
      help='Words to use for evaluatioN',
      nargs='+',
      type=str,
  )
  parser.add_argument(
      '--eval-data-paths',
      help='TFRecord files for evaluation',
      required=True
  )
  parser.add_argument(
      '--batch-size',
      help='Batch size for TFRecord files',
      type=int,
      default=512
  )
  parser.add_argument(
      '--num-epochs',
      help='Number of epochs for training',
      type=int,
      default=1
  )
  parser.add_argument(
      '--hyper-params',
      help='JSON dict of hyperparameters',
      type=json.loads
  )
  args = parser.parse_args()
  learn_runner.run(make_experiment_fn(args), args.output_path)
