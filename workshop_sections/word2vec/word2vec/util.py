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

tf.logging.set_verbosity(tf.logging.INFO)


def window_input_producer(tensor, window_size, capacity=32, num_epochs=None):
  num_windows = tensor.get_shape().dims[0].value - window_size
  if num_windows <= 0:
    raise ValueError('Provided tensor is not large enough for a window')
  range_queue = tf.train.range_input_producer(
      tf.constant(num_windows, dtype=tf.int32, shape=[]),
      shuffle=False,
      capacity=capacity,
      num_epochs=num_epochs
  )
  index = range_queue.dequeue()
  window = tensor[index:index + window_size]
  queue = tf.FIFOQueue(capacity=capacity,
                       dtypes=[tensor.dtype.base_dtype],
                       shapes=[window_size])

  enq = queue.enqueue(window)
  tf.train.add_queue_runner(
      tf.train.QueueRunner(queue, [enq])
  )

  return queue


def skipgrams(word_tensor, num_skips, skip_window, num_epochs=None):
  window_size = 2 * skip_window + 1
  indices = range(window_size)
  del indices[skip_window]

  indices_tensor = tf.random_shuffle(
      tf.constant(indices, dtype=tf.int32)
  )[:num_skips]

  windows = window_input_producer(word_tensor, window_size, num_epochs=num_epochs)
  window = windows.dequeue()

  targets = tf.tile([window[skip_window]], [num_skips])
  contexts = tf.gather(window, indices_tensor)
  return targets, contexts


def build_vocab(word_tensor, vocab_size):
  unique, idx = tf.unique(word_tensor)
  counts_one_hot = tf.one_hot(
      idx,
      idx[tf.to_int32(tf.argmax(idx, 0))] + 1,
      dtype=tf.int32
  )
  counts = tf.reduce_sum(counts_one_hot, 0)
  _, indices = tf.nn.top_k(counts, k=vocab_size)
  return tf.gather(unique, indices)


def make_input_fn(text_file,
                  batch_size,
                  num_skips,
                  skip_window,
                  vocab_size,
                  num_epochs=None):
  def _input_fn():
    with tf.name_scope('input'):
      corpus = tf.parse_tensor(tf.read_file(text_file), tf.string)

      word_tensor = tf.py_func(
          lambda t: np.array(nltk.word_tokenize(t)),
          [corpus],
          tf.string,
          stateful=False,
          name='tokenize'
      )
      index = build_vocab(word_tensor, vocab_size)

      targets, contexts = skipgrams(
          word_tensor,
          num_skips,
          skip_window,
          num_epochs=num_epochs
      )

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
