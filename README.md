
# TensorFlow workshop materials

This repo contains materials for use in TensorFlow workshops.

Contributions are not currently accepted.  This is not an official Google product.

<!---
[** add note re: TF versions tested with **].
-->

## Getting Started

- [Building a small starter TensorFlow graph](workshop_sections/getting_started/starter_tf_graph/README.md)
- [XOR: A minimal training example](workshop_sections/getting_started/xor/README.md)
- A [LinearRegressor example](workshop_sections/linear_regressor_datasets) that uses Datasets.

## [The high-level TensorFlow APIs, via MNIST & 'fashion MNIST'](workshop_sections/high_level_APIs)

- [Using TensorFlow's high-level APIs to build classifiers, and introducing TensorBoard](workshop_sections/high_level_APIs/mnist_estimator.ipynb).
- [An example of using Keras with TensorFlow eager mode, on the'Fashion MNIST' dataset](workshop_sections/high_level_APIs/mnist_eager_keras.ipynb): This notebook shows an example of using Keras with TensorFlow eager mode, on the 'Fashion MNIST' dataset. This notebook requires TensorFlow >=1.7.
- [Building Custom `Estimator`s for a version of MNIST that uses CNNs](workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/README.md), using either TensorFlow or [Keras](https://keras.io/) layers.


## 'Wide & Deep'

- [Using a tf.estimator to train a 'Wide & Deep' model](workshop_sections/wide_n_deep/README.md). The example highlights use of [`tf.feature_columns`](https://www.tensorflow.org/versions/master/get_started/feature_columns), which are intermediaries between raw data and Estimators, enabling you to transform a diverse range of raw data into formats that Estimators can use, and allowing easy experimentation.
It also includes the use of [**`tf.estimator.train_and_evaluate`**](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) and [**Datasets**](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).


## Extras

- [Transfer learning: using a trained model to 'bootstrap' learning new classifications](transfer_learning/README.md) [using Cloud ML Engine](workshop_sections/transfer_learning/cloudml). This example still works (and is fun to play with), but uses TensorFlow v1.2.  So it doesn't demonstrate current best practices.

- **(probably outdated)** [Building a word2vec model using a Custom Estimator, and exploring the learned embeddings](workshop_sections/word2vec/README.md). Introducing [TFRecords](https://www.tensorflow.org/api_guides/python/python_io).

In addition, there is an [extras](workshop_sections/extras/README.md) directory, that contains some older labs not currently used in this workshop (& which may not necessarily run with the latest version of TF), but which may be of interest.

