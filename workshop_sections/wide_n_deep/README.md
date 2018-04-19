
# the Wide & Deep Estimator, with tf.feature_columns, tf.data, and tf.estimator.train_and_evaluate

The examples in this directory use the "Wide & Deep" [prebuilt Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier).
Wide and deep models use a deep neural net (DNN) to learn high level abstractions about complex features or interactions between such features. These models then combine the outputs from the DNN with a [linear regression](https://en.wikipedia.org/wiki/Linear_regression) performed on simpler features. This provides a balance between power and speed that is effective on many structured data problems.
You can read more about this model and its use [here](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html). 

These examples highlight use of [`tf.feature_columns`](https://www.tensorflow.org/versions/master/get_started/feature_columns), which are intermediaries between raw data and Estimators. Feature columns are very rich, enabling you to transform a diverse range of raw data into formats that Estimators can use, allowing easy experimentation.

The example notebook in the [tf_train_and_evaluate](tf_train_and_evaluate) directory also includes the use of
[**`tf.estimator.train_and_evaluate`**](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) and [**Datasets**](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).  


## Installation

Install [TensorFlow](https://www.tensorflow.org/install) and [Jupyter](https://jupyter.org/install.html). 
Your TensorFlow version must be >=1.4. Ideally, use the latest version.

Additionally, the notebook in the [tf_train_and_evaluate](tf_train_and_evaluate) directory includes instructions for optionally running training and prediction on Google Cloud Platform (GCP) [Cloud ML Engine](https://cloud.google.com/ml-engine).  Follow the instructions in that directory to set up a GCP project.


# # About the dataset

We will use a version of the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). This data was extracted from the 1994 US Census by Barry Becker. 
The prediction task is to determine whether a person makes over 50K a year.
