# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


tf.logging.set_verbosity(tf.logging.ERROR)

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status",
                       "occupation", "relationship", "race", "gender",
                       "native_country"]

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

# Our feature columns are everything except fnlwgt, which is not used, and
# income_bracket, which is our label column
FEATURE_COLUMNS = list(set(COLUMNS) - set(["fnlwgt", "income_bracket"]))


def generate_input_fn(filename):
  def _input_fn():
    BATCH_SIZE = 40
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)

    record_defaults = [[0], [" "], [0], [" "], [0],
                       [" "], [" "], [" "], [" "], [" "],
                       [0], [0], [0], [" "], [" "]]
    columns = tf.decode_csv(
        value, record_defaults=record_defaults)

    features = dict(zip(COLUMNS, columns))

    # save our label
    income_bracket = features.pop('income_bracket')
    
    # remove the fnlwgt key, which is not used
    features.pop('fnlwgt', 'fnlwgt key not found')

    # works in 0.12 only
    for feature_name in CATEGORICAL_COLUMNS:
      features[feature_name] = tf.expand_dims(features[feature_name], -1)

    income_int = tf.to_int32(tf.equal(income_bracket, " >50K"))

    return features, income_int

  return _input_fn


def build_feature_cols():
  # Sparse base columns.
  gender = layers.sparse_column_with_keys(
            column_name="gender",
            keys=["female", "male"])
  race = layers.sparse_column_with_keys(
            column_name="race",
            keys=["Amer-Indian-Eskimo",
                  "Asian-Pac-Islander",
                  "Black", "Other",
                  "White"])

  education = layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  marital_status = layers.sparse_column_with_hash_bucket(
      "marital_status", hash_bucket_size=100)
  relationship = layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # Continuous base columns.
  age = layers.real_valued_column("age")
  education_num = layers.real_valued_column("education_num")
  capital_gain = layers.real_valued_column("capital_gain")
  capital_loss = layers.real_valued_column("capital_loss")
  hours_per_week = layers.real_valued_column("hours_per_week")

  # Transformations.
  age_buckets = layers.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  education_occupation = layers.crossed_column(
      [education, occupation], hash_bucket_size=int(1e4))
  age_race_occupation = layers.crossed_column(
      [age_buckets, race, occupation], hash_bucket_size=int(1e6))
  country_occupation = layers.crossed_column(
      [native_country, occupation], hash_bucket_size=int(1e4))

  # Wide columns and deep columns.
  wide_columns = [gender, native_country, education, 
                  occupation, workclass, race, 
                  marital_status, relationship, 
                  age_buckets,
                  education_occupation, 
                  age_race_occupation,
                  country_occupation]

  deep_columns = [
      layers.embedding_column(gender, dimension=8),
      layers.embedding_column(native_country, dimension=8),
      layers.embedding_column(education, dimension=8),
      layers.embedding_column(occupation, dimension=8),
      layers.embedding_column(workclass, dimension=8),
      layers.embedding_column(race, dimension=8),
      layers.embedding_column(marital_status, dimension=8),
      layers.embedding_column(relationship, dimension=8),
      # layers.embedding_column(age_buckets, dimension=8),
      layers.embedding_column(education_occupation, dimension=8),
      layers.embedding_column(age_race_occupation, dimension=8),
      layers.embedding_column(country_occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  return wide_columns, deep_columns


def build_model(model_type, model_dir, wide_columns, deep_columns):
  m = None
  deep_layers = [100, 70, 50, 25]

  # Linear Classifier
  if model_type == 'WIDE':
    m = tf.contrib.learn.LinearClassifier(
      model_dir=model_dir, 
      feature_columns=wide_columns)

  # Deep Neural Net Classifier
  elif model_type == 'DEEP':
    m = tf.contrib.learn.DNNClassifier(
      model_dir=model_dir,
      feature_columns=deep_columns,
      hidden_units=deep_layers)

  # Combined Linear and Deep Classifier
  elif model_type == 'WIDE_AND_DEEP':
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=deep_layers)

  print('created {} model'.format(model_type))

  return m


def build_estimator(model_type='WIDE_AND_DEEP', model_dir=None):
  if model_dir is None:
    model_dir = 'models/model_' + model_type + '_' + str(int(time.time()))
    print("Model directory = %s" % model_dir)

  wide_columns, deep_columns = build_feature_cols()
  m = build_model(model_type, model_dir, wide_columns, deep_columns)
  print('estimator built')
  return m


# All categorical columns are strings for this dataset
def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32


"""
  This function maps input columns (feature_placeholders) to 
  tensors that can be inputted into the graph 
  (similar in purpose to the output of our input functions)
  In this particular case, we need to accomodate the sparse fields (strings)
  so we have to do a slight modification to expand their dimensions, 
  just like in the input functions
"""
def serving_input_fn():
    feature_placeholders = {
        column: tf.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }
    # DNNCombinedLinearClassifier expects rank 2 Tensors, 
    # but inputs should be rank 1, so that we can provide 
    # scalars to the server
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    
    return input_fn_utils.InputFnOps(
        features, # input into graph
        None,
        feature_placeholders # tensor input converted from request 
    )


def generate_experiment(output_dir, train_file, test_file, model_type):
  def _experiment_fn(output_dir):
    train_input_fn = generate_input_fn(train_file)
    eval_input_fn = generate_input_fn(test_file)
    my_model = build_estimator(model_type=model_type, 
                               model_dir=output_dir)

    experiment = tf.contrib.learn.Experiment(
      my_model,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=1000
      ,
      export_strategies=[saved_model_export_utils.make_export_strategy(
        serving_input_fn,
        default_output_alternative_key=None
      )]
    )
    return experiment

  return _experiment_fn


def train_and_eval(job_dir=None, model_type='WIDE_AND_DEEP'):
  print("Begin training and evaluation")

  # if local eval and no args passed, default
  if job_dir is None: job_dir = 'models/' 

  # Ensure path has a '/' at the end
  if job_dir[-1] != '/': job_dir += '/'

  gcs_base = 'https://storage.googleapis.com/'
  gcs_path = 'cloudml-public/census/data/'
  trainfile = 'adult.data.csv'
  testfile  = 'adult.test.csv'
  local_path = 'dataset_files'
  train_file = base.maybe_download(
    trainfile, local_path, gcs_base + gcs_path + trainfile)
  test_file = base.maybe_download(
    testfile, local_path, gcs_base + gcs_path + testfile)

  training_mode = 'learn_runner'
  train_steps = 1000
  test_steps = 100

  model_dir = job_dir + 'model_' + model_type + '_' + str(int(time.time()))
  print("Saving model checkpoints to " + model_dir)
  export_dir = model_dir + '/exports'

  # Manually train and export model
  if training_mode == 'manual':
    # In this function, editing below here is unlikely to be needed
    m = build_estimator(model_type, model_dir)

    m.fit(input_fn=generate_input_fn(train_file), steps=train_steps)
    print('fit done')

    results = m.evaluate(input_fn=generate_input_fn(test_file), steps=test_steps)
    print('evaluate done')

    print('Accuracy: %s' % results['accuracy'])

    export_folder = m.export_savedmodel(
      export_dir_base = export_dir,
      input_fn=serving_input_fn
    )

    print('Model exported to ' + export_dir)


  elif training_mode == 'learn_runner':
    # use learn_runner
    experiment_fn = generate_experiment(
      model_dir, train_file, test_file, model_type)

    metrics, output_folder = learn_runner.run(experiment_fn, model_dir)

    print('Accuracy: {}'.format(metrics['accuracy']))
    print('Model exported to {}'.format(output_folder))


def version_is_less_than(a, b):
    a_parts = a.split('.')
    b_parts = b.split('.')
    
    for i in range(len(a_parts)):
        if int(a_parts[i]) < int(b_parts[i]):
            print('{} < {}, version_is_less_than() returning False'.format(
              a_parts[i], b_parts[i]))
            return True
    return False

def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
      return tf.string
    else:
      return tf.float32

def serving_input_fn():
  feature_placeholders = {
    column: tf.placeholder(column_to_dtype(column), [None])
      for column in FEATURE_COLUMNS
  }
  # DNNCombinedLinearClassifier expects rank 2 Tensors, but inputs should be
  # rank 1, so that we can provide scalars to the server
  features = {
    key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
  }
  
  return input_fn_utils.InputFnOps(
    features, # input into graph
    None,
    feature_placeholders # tensor input converted from request 
  )
      
def get_arg_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=False
  )

  parser.add_argument(
      '--model-type',
      help='Whether to run WIDE, DEEP, or WIDE_AND_DEEP model. Default is WIDE_AND_DEEP',
      required=False,
      default='WIDE_AND_DEEP'
    )

  return parser

if __name__ == "__main__":
  print("TensorFlow version {}".format(tf.__version__))
  required_tf_version = '1.0.0'
  if version_is_less_than(tf.__version__ , required_tf_version):
    raise ValueError('This code requires tensorflow >= ' + str(required_tf_version))

  parser = get_arg_parser()
  args = parser.parse_args()
  train_and_eval(args.job_dir, model_type=args.model_type)
