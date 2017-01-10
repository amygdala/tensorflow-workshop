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

import argparse

import nltk
import numpy as np
import tensorflow as tf


def build_vocab(word_tensor, vocab_size):
  unique, idx = tf.unique(word_tensor)

  counts = tf.foldl(
      lambda counts, item: counts + tf.one_hot(
          tf.reshape(item, [-1]),
          tf.shape(unique)[0],
          dtype=tf.int32)[0],
      idx,
      initializer=tf.zeros_like(unique, dtype=tf.int32),
      back_prop=False
  )
  _, indices = tf.nn.top_k(counts, k=vocab_size)
  return tf.gather(unique, indices)

def build_string_index(word_array, vocab_size=2 ** 15):
  unique, counts = np.unique(word_array, return_counts=True)

  sort_unique = np.argsort(counts)
  sorted_counts = counts[sort_unique][::-1][:vocab_size - 1]
  unique_sorted = unique[sort_unique][::-1][:vocab_size - 1]

  return unique_sorted, sorted_counts, word_array

def prepare_data_files(infile, outfile, vocab_size):
  with open(infile, 'r') as reader:
    words = np.array(nltk.word_tokenize(reader.read()))

  train_eval_split = int(len(words) * .9)
  train_words = words[:train_eval_split]
  eval_words = words[train_eval_split:]
  with open('{}-train.pb2'.format(outfile), 'wb') as writer:
    writer.write(tf.contrib.util.make_tensor_proto(
        train_words).SerializeToString())

  with open('{}-eval.pb2'.format(outfile), 'wb') as writer:
    writer.write(tf.contrib.util.make_tensor_proto(
        eval_words).SerializeToString())

  index, _, _ = build_string_index(words, vocab_size)

  with open('{}-index.pb2'.format(outfile), 'wb') as writer:
    writer.write(tf.contrib.util.make_tensor_proto(
        index).SerializeToString())


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--vocab-size', type=int, default=2**15)
  parser.add_argument('--text-file', type=str)
  parser.add_argument('--output-path', type=str)
  args = parser.parse_args()
  prepare_data_files(args.text_file, args.output_path, args.vocab_size)
