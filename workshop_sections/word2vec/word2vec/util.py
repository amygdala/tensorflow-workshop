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
from __future__ import division
from __future__ import print_function

import multiprocessing

import nltk
import numpy as np
import tensorflow as tf


def build_string_index(string, vocab_size=2 ** 15):
  word_array = np.array(nltk.word_tokenize(string))

  unique, counts = np.unique(word_array, return_counts=True)

  sort_unique = np.argsort(counts)
  sorted_counts = counts[sort_unique][::-1][:vocab_size - 1]
  unique_sorted = unique[sort_unique][::-1][:vocab_size - 1]

  return unique_sorted, sorted_counts, word_array


def rolling_window(tensor, dtype, shape, capacity=None):
  with tf.name_scope('rolling_window'):
    window_size = shape[0]
    if capacity is None:
      capacity = shape[0] * 2 + 1

    q = tf.FIFOQueue(capacity, [dtype], shapes=[shape[1:]])
    enqueue = q.enqueue_many(tensor)
    tf.train.add_queue_runner(
        tf.train.QueueRunner(queue=q, enqueue_ops=[enqueue])
    )

    # Pad first element as it will be immediately overwritten
    window_initial_value = q.dequeue_many(window_size)

    window = tf.Variable(
        window_initial_value,
        expected_shape=shape,
        trainable=False,
        collections=[],
        name='window'
    )
    oldest_pos = tf.Variable(
        0,
        expected_shape=[],
        trainable=False,
        dtype=tf.int32,
        collections=[],
        name='oldest_pos'
    )

    oldest_pos_init = tf.cond(tf.is_variable_initialized(oldest_pos),
                              lambda: tf.assign(oldest_pos, oldest_pos + 1 % window_size),
                              lambda: _init_and_return_ref(oldest_pos))

    window_init = tf.cond(tf.is_variable_initialized(window),
                          lambda: window[oldest_pos_init].assign(q.dequeue()),
                          lambda: _init_and_return_ref(window))

    return tf.concat(0, [
        window_init[oldest_pos_init+1:],
        window_init[:oldest_pos_init+1]
    ])


def _init_and_return_ref(ref):
    return tf.tuple([ref], control_inputs=[ref.initializer])

def skipgrams(word_tensor, num_skips, skip_window):
  window = rolling_window(
      tf.expand_dims(word_tensor, axis=1),
      tf.string,
      [2*skip_window + 1, 1]
  )
  target = window[skip_window]
  contexts = tf.random_shuffle(
      tf.concat(0, [
          window[:skip_window],
          window[skip_window+1:]
      ])
  )[:num_skips]

  targets = tf.tile(target, [num_skips])
  return targets, contexts


def write_index_to_file(index, filename):
  with open(filename, 'wb') as writer:
    writer.write(tf.contrib.util.make_tensor_proto(index).SerializeToString())


def make_input_fn(filenames,
                  batch_size,
                  num_skips,
                  skip_window,
                  index_file,
                  num_epochs=None):
  def _input_fn():
    with tf.name_scope('input'):
      index = tf.parse_tensor(tf.read_file(index_file), tf.string)
      filename_queue = tf.train.string_input_producer(
          filenames, num_epochs=num_epochs)
      reader = tf.TextLineReader()
      _, string_tensor = reader.read(filename_queue)

      word_tensor = tf.py_func(
          lambda t: np.array(nltk.word_tokenize(t)),
          [string_tensor],
          tf.string,
          stateful=False,
          name='tokenize'
      )

      targets, contexts = skipgrams(word_tensor, num_skips, skip_window)

      thread_count = multiprocessing.cpu_count()

      # The minimum number of instances in a queue from which examples are
      # drawn randomly. The larger this number, the more randomness at the
      # expense of higher memory requirements.
      min_after_dequeue = batch_size * 10

      # When batching data, the queue's capacity will be larger than the
      # batch_size by some factor. The recommended formula is (num_threads +
      # a small safety margin). For now, we use a single thread for reading,
      # so this can be small.
      queue_size_multiplier = thread_count + 3

      target_batch, context_batch = tf.train.shuffle_batch(
          [targets, contexts],
          batch_size=batch_size,
          num_threads=thread_count,
          capacity=min_after_dequeue + queue_size_multiplier * batch_size,
          min_after_dequeue=min_after_dequeue,
          enqueue_many=True
      )

      return {
          'targets': target_batch,
          'index': index
      }, context_batch

  return _input_fn
