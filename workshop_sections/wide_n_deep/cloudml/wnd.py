from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR);

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
  "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
  "hours_per_week", "native_country", "income_bracket"]

COLUMNS_KEEP = ["age", "workclass", "education", "education_num", "marital_status",
  "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
  "hours_per_week", "native_country", "income_bracket"]


def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                     keys=["female", "male"])
  race = tf.contrib.layers.sparse_column_with_keys(column_name="race",
                                                   keys=["Amer-Indian-Eskimo",
                                                         "Asian-Pac-Islander",
                                                         "Black", "Other",
                                                         "White"])

  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
      "marital_status", hash_bucket_size=100)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")


  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                boundaries=[ 18, 25, 30, 35, 40, 45, 50, 55, 60, 65 ])
  education_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
  age_race_occupation = tf.contrib.layers.crossed_column( [age_buckets, race, occupation], hash_bucket_size=int(1e6))
  country_occupation = tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4))



  # Wide columns and deep columns.
  wide_columns = [gender, native_country,
          education, occupation, workclass,
          marital_status, relationship,
          age_buckets, education_occupation,
          age_race_occupation, country_occupation]

  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marital_status,
                                         dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(race, dimension=8),
      tf.contrib.layers.embedding_column(native_country,
                                         dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  # m = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)

  #  m = tf.contrib.learn.DNNClassifier(
  #         model_dir=model_dir,
  #         feature_columns=deep_columns,
  #         hidden_units=[100, 50])

  m = tf.contrib.learn.DNNLinearCombinedClassifier(
         model_dir=model_dir,
         linear_feature_columns=wide_columns,
         dnn_feature_columns=deep_columns,
         dnn_hidden_units=[100, 50])

  return m

# def read_parser(t):
#   tf.Print(t, [t])
#   print('example tensor',t)
#   return t

def generate_input_fn(filename):
  def _input_fn():
    BATCH_SIZE = 40
    filename_queue = tf.train.string_input_producer([filename])
        # "gs://run-wild-ml/adult.data",
        # "gs://run-wild-ml/adult.test"])
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)


    # batch_reader = tf.TextLineReader()
    # value = tf.contrib.learn.read_batch_examples(
    #   file_pattern=["gs://run-wild-ml/adult.data",
    #                 "gs://run-wild-ml/adult.test"],
    #   batch_size=BATCH_SIZE,
    #   reader = tf.TextLineReader,
    #   num_epochs=1,
    #   parse_fn=read_parser
    # )

    record_defaults = [[0], [" "], [0], [" "], [0],
                    [" "], [" "], [" "], [" "], [" "],
                    [0], [0], [0], [" "], [" "]]
    # shape of each col is BATCH_SIZE
    # 39, State-gov, 77516, Bachelors, 13,
    # Never-married, Adm-clerical, Not-in-family, White, Male,
    # 2174, 0, 40, United-States, <=50K

    # TODO: use indexing cleverness to remove fnlwgt to clean things up
    # age, workclass, fnlwgt, education, education_num, marital_status, occupation, \
    # relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country, \
    # income_bracket
    columns = tf.decode_csv(
      value, record_defaults=record_defaults)

    features, income_bracket = dict(zip(COLUMNS, columns[:-1])), columns[-1]
    # feature_cols = tf.decode_csv(value, record_defaults=record_defaults)

    # remove the fnlwgt key
    # print('attempt fnlwgt key removal: ')
    print(features.pop('fnlwgt', 'fnlwgt key not found'))

    # feature_val = tf.pack([
    #     age, workclass, education, education_num, marital_status, occupation, \
    #     relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country
    #   ], axis=1)
    # print(feature_val)

    # Doing it with tensors
    for feature_name in CATEGORICAL_COLUMNS:
      # features[feature_name] = tf.SparseTensor(
      #   indices=tf.expand_dims(
      #     tf.range(0,
      #       # tf.to_int64(
      #         tf.shape(features[feature_name])[0]#)
      #       ), -1),
      #   values=features[feature_name],
      #   shape=tf.concat(0, [tf.shape(features[feature_name]), [1]])
      # )

      # feature_shape = tf.shape(features[feature_name], name="features_shape", out_type=tf.int64)[0]
      # feature_range = tf.range(tf.to_int64(0), feature_shape)

      # features[feature_name] = tf.SparseTensor(
      #   indices=tf.pack([feature_range, tf.zeros_like(feature_range, dtype=tf.int64)], axis=1),
      #   values = features[feature_name],
      #   shape  =[feature_shape, 1]
      # )

    # works in 0.12 only
    # for feature_name in CATEGORICAL_COLUMNS:
      features[feature_name] = tf.expand_dims(features[feature_name], -1)


      # Doing it with python
      # features[feature_name] = tf.SparseTensor(
      #   indices=[[i, 0] for i in range(BATCH_SIZE)],
      #   values=features[feature_name],
      #   shape=[BATCH_SIZE, 1]
      # )


    # features_array = [
    #     age, workclass, education, education_num, marital_status, occupation, \
    #     relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country
    #   ]
#     features = dict(zip(COLUMNS_KEEP, features_array))

#     print(features)
# #     df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    income_int = tf.to_int32(tf.equal(income_bracket, " >50K"))
    # income_int = tf.to_int32(tf.equal(income_bracket, tf.constant(' >50K', dtype=tf.string)))

    # print(income_int)

    # features = dict(zip(COLUMNS, tf.constant(feature_cols[:-1])))
    # print(feature_cols[:-1])
    # features = dict(zip(COLUMNS, feature_cols[:-1]))

    return features, income_int
    # return feature_val, feature_cols[-1]

  return _input_fn


def train_and_eval():

  train_file = "gs://run-wild-ml/adult.data"
  test_file = "gs://run-wild-ml/adult.test"
  train_steps = 1000

  # model_dir = tempfile.mkdtemp()
  model_dir = 'models/model_' + str(int(time.time()))
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  print('estimator built')

  m.fit(input_fn=generate_input_fn(train_file), steps=train_steps)
  print('fit done')

  results = m.evaluate(input_fn=generate_input_fn(test_file), steps=1)
  print('evaluate done')

  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

  # print('accuracy: ' + results['accuracy'])


if __name__ == "__main__":
  print("TensorFlow version %s" % (tf.__version__))
  train_and_eval()
