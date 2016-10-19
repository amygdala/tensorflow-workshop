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

import math

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys


GCS_SKIPGRAMS = [
    'gs://oscon-tf-workshop-materials/skipgrams/batches-{}.pb2'.format(i)
    for i in range(4)
]


def make_model_fn(index,
                  vocab_counts,
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
            index_tensor, tf.constant(range(vocab_size - 1), dtype=tf.int64)
        ),
        vocab_size - 1
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

        sampled_words = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=context_indices,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocab_size,
            distortion=0.75,
            unigrams=vocab_counts + [1]
        )
        loss = tf.reduce_mean(tf.nn.nce_loss(
            nce_weights,
            nce_biases,
            embedded,
            context_indices,
            num_sampled,
            vocab_size,
            sampled_values=sampled_words
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
        index_tensor = tf.concat(0, [index_tensor, tf.constant(['UNK'])])
        prediction_dict['predictions'] = tf.gather(index_tensor, predictions)

      return prediction_dict, loss, train_op
  return _make_model


def _rolling_window(a, window):
  shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
  strides = a.strides + (a.strides[-1],)
  return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def to_skipgrams(window, skip_window, num_skips):
  contexts = np.random.choice(np.concatenate(
      (window[:skip_window], window[skip_window + 1:])
  ), size=num_skips, replace=False)
  targets = np.repeat(window[skip_window], num_skips)
  return np.concatenate((targets, contexts))


def generate_batches(word_array, num_skips=2, skip_window=1):
  assert num_skips <= 2 * skip_window
  span = 2 * skip_window + 1

  span_windows = _rolling_window(word_array, span)
  batches = np.apply_along_axis(
    to_skipgrams, 1, span_windows, skip_window, num_skips)
  # Separate targets and contexts
  batches_sep = np.reshape(batches, (-1, 2, num_skips))
  # Gather targets and contexts
  batches_gathered = np.transpose(batches_sep, (1, 0, 2))
  # Squash targets and contexts
  batches_squashed = np.reshape(batches_gathered, (2, -1))
  return batches_squashed[0], batches_squashed[1]

def build_string_index(string, vocab_size=2 ** 15):
  word_array = np.array(nltk.word_tokenize(string))

  unique, counts = np.unique(word_array, return_counts=True)

  sort_unique = np.argsort(counts)
  sorted_counts = counts[sort_unique][::-1][:vocab_size - 1]
  unique_sorted = unique[sort_unique][::-1][:vocab_size - 1]

  return unique_sorted, sorted_counts, word_array


def input_from_files(filenames, batch_size, num_epochs=1):
  def _make_input_fn():
    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(
          filenames, num_epochs=num_epochs)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)

      words = tf.parse_single_example(
          serialized_example,
          {
              'target_words': tf.FixedLenFeature([batch_size], tf.string),
              'context_words': tf.FixedLenFeature([batch_size], tf.string)
          }
      )
      return tf.expand_dims(words['target_words'], 1), words['context_words']
  return _make_input_fn


def write_batches_to_file(filename,
                          batch_size,
                          word_array,
                          num_skips=8,
                          skip_window=4,
                          num_shards=4):
    span = 2 * skip_window + 1
    span_windows = _rolling_window(word_array, span)
    span_batch_size = batch_size // num_skips
    span_windows_len = (len(span_windows) // span_batch_size) * span_batch_size
    span_windows_trunc = span_windows[:span_windows_len]
    window_batches = np.reshape(
        span_windows_trunc, (-1, span_batch_size, span))

    shard_size = len(window_batches) // num_shards
    for shard, index in enumerate(range(0, len(window_batches), shard_size)):
      shard_file = '{}-{}.pb2'.format(filename, shard)
      with tf.python_io.TFRecordWriter(shard_file) as writer:
        for windows in window_batches[index:index+shard_size]:
          batches = np.apply_along_axis(
              to_skipgrams, 1, windows, skip_window, num_skips)
          # Separate targets and contexts
          batches_sep = np.reshape(batches, (-1, 2, num_skips))
          # Gather targets and contexts
          batches_gathered = np.transpose(batches_sep, (1, 0, 2))
          # Squash targets and contexts
          batches_squashed = np.reshape(batches_gathered, (2, -1))

          writer.write(
              tf.train.Example(features=tf.train.Features(feature={
                  'target_words': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=batches_squashed[0])),
                  'context_words': tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=batches_squashed[1]))
              })).SerializeToString()
          )
