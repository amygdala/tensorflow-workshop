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
"""Reusable input_fn factory."""

import multiprocessing

import tensorflow as tf


def make_input_fn(files,
                  example_parser,
                  batch_size,
                  shuffle=True,
                  num_epochs=None,
                  reader_options=None):
  def _input_fn():
    """Creates readers and queues for reading example protos."""
    thread_count = multiprocessing.cpu_count()

    # The minimum number of instances in a queue from which examples are drawn
    # randomly. The larger this number, the more randomness at the expense of
    # higher memory requirements.
    min_after_dequeue = 1000

    # When batching data, the queue's capacity will be larger than the
    # batch_size by some factor. The recommended formula is (num_threads +
    # a small safety margin). For now, we use a single thread for reading,
    # so this can be small.
    queue_size_multiplier = thread_count + 3

    # Build a queue of the filenames to be read.
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=shuffle)

    example_id, encoded_examples = tf.TFRecordReader(
        options=reader_options).read_up_to(filename_queue, batch_size)

    features, targets = example_parser(encoded_examples)
    if shuffle:
      capacity = min_after_dequeue + queue_size_multiplier * batch_size
      feature_batch, target_batch = tf.train.shuffle_batch(
          [features, targets],
          batch_size,
          capacity,
          min_after_dequeue,
          enqueue_many=True,
          num_threads=thread_count)

    else:
      capacity = queue_size_multiplier * batch_size
      feature_batch, target_batch = tf.train.batch(
          [features, targets],
          batch_size,
          capacity=capacity,
          enqueue_many=True,
          num_threads=thread_count)

    return feature_batch, target_batch
  return _input_fn
