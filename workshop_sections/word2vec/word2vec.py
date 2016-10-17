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
    'gs://oscon-tf-workshop-materials/skipgrams/shard-0.pb2',
    'gs://oscon-tf-workshop-materials/skipgrams/shard-17007616.pb2',
    'gs://oscon-tf-workshop-materials/skipgrams/shard-25511424.pb2',
    'gs://oscon-tf-workshop-materials/skipgrams/shard-8503808.pb2',
]


def make_model_fn(num_partitions=1,
                  embedding_size=128,
                  vocab_size=2 ** 15,
                  num_sim=8,
                  num_sampled=64,
                  learning_rate=0.1):
  def _make_model(target_words, context_words, mode):
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

      predictions, loss, train_op = ({}, None, None)
      # tf.contrib.learn.Estimator.fit adds an addition dimension to input
      target_words_squeezed = tf.squeeze(target_words, squeeze_dims=[1])

      if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
        embedded = tf.nn.embedding_lookup(embeddings, target_words_squeezed)
        target_words = tf.expand_dims(context_words, 1)
        loss = tf.reduce_mean(tf.nn.nce_loss(
            nce_weights,
            nce_biases,
            embedded,
            target_words,
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
            normalized_embeddings, target_words_squeezed)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
        predictions['values'], predictions['predictions'] = tf.nn.top_k(
            similarity, sorted=True, k=num_sim)

      return predictions, loss, train_op
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


def generate_batches(word_indices, num_skips=2, skip_window=1):
  assert num_skips <= 2 * skip_window
  span = 2 * skip_window + 1

  span_windows = _rolling_window(word_indices, span)
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

  unique, inverse, counts = np.unique(
    word_array, return_inverse=True, return_counts=True)

  max_index = len(unique) - 1
  sort_unique = np.argsort(counts)
  shuffle_idx = np.searchsorted(counts[sort_unique], counts)
  unique_sorted = unique[sort_unique][::-1]

  index = np.concatenate((np.array(['UNK']), unique_sorted[:vocab_size - 1]))

  indices = max_index - shuffle_idx + 1
  indices = np.where(
    indices > vocab_size - 1,
    np.zeros(len(indices), dtype=np.int8),
    indices
  )

  word_indices = indices[inverse]
  return index, word_indices


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
              'target_words': tf.FixedLenFeature([batch_size], tf.int64),
              'context_words': tf.FixedLenFeature([batch_size], tf.int64)
          }
      )
      return tf.expand_dims(words['target_words'], 1), words['context_words']
  return _make_input_fn

def write_batches_to_file(filename,
                          batch_size,
                          targets,
                          contexts,
                          num_shards=4):
  
  end = len(targets) - (len(targets) % batch_size)
  shard_size = end // num_shards
  for shard in range(0, end, end // num_shards):
    with tf.python_io.TFRecordWriter('{}-{}.pb2'.format(filename, shard)) as writer:
      for i in range(shard, shard+shard_size, batch_size):
        writer.write(
            tf.train.Example(features=tf.train.Features(feature={
                'target_words': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=targets[i:i+batch_size])),
                'context_words': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=contexts[i:i+batch_size]))
            })).SerializeToString()
        )
