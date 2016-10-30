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

import tensorflow as tf
from tensorflow.contrib.learn import Experiment, Estimator
from tensorflow.contrib.learn.python.learn import learn_runner

import model
import util


tf.logging.set_verbosity(tf.logging.INFO)


def model_args(parser):
  group = parser.add_argument_group(title='Model Arguments')
  group.add_argument('--reference-words', nargs='*', type=str)
  group.add_argument('--num-partitions', default=1, type=int)
  group.add_argument('--embedding-size', default=128, type=int)
  group.add_argument('--vocab-size', default=2 ** 15, type=int)
  group.add_argument('--num-sim', default=8, type=int)
  group.add_argument('--num-sampled', default=64, type=int)
  group.add_argument('--learning-rate', default=0.1, type=float)
  return group


def make_experiment_fn(args):
  train_input_fn = util.make_input_fn(
      args.train_data_paths,
      args.batch_size,
      args.index_file,
      num_epochs=args.num_epochs
  )
  eval_input_fn = util.make_input_fn(
      args.eval_data_paths,
      args.batch_size,
      args.index_file,
      num_epochs=args.num_epochs
  )
  def experiment_fn(output_dir):
    return Experiment(
        Estimator(
            model_fn=model.make_model_fn(args),
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
      nargs='+',
      type=str
  )
  parser.add_argument(
      '--output-path',
      help='GCS path to output files',
      required=True
  )
  parser.add_argument(
      '--eval-data-paths',
      help='TFRecord files for evaluation',
      nargs='+',
      type=str
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
  model_args(parser)
  args = parser.parse_args()
  learn_runner.run(make_experiment_fn(args), args.output_path)
