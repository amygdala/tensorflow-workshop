
# TensorFlow workshop materials

This repo contains materials for use in a TensorFlow workshop.  
Contributions are not currently accepted.  This is not an official Google product.

This document points to more information for each workshop lab.  

<!---
[** add note re: TF versions tested with **].
-->

## Getting Started

- [Building a small starter TensorFlow graph](workshop_sections/getting_started/starter_tf_graph/README.md)
- [XOR: A minimal training example](workshop_sections/getting_started/xor/README.md)
- A [LinearRegressor example](workshop_sections/linear_regressor_datasets) that uses Datasets.

## The MNIST (& 'fashion MNIST') series

- [Introducing MNIST, and building a simple linear classifier in TensorFlow](workshop_sections/mnist_series/01_README_mnist_simple.md).
- [Using TensorFlow's high-level APIs to build an MNIST DNN Classifier, and introducing TensorBoard](workshop_sections/mnist_series/02_README_mnist_tflearn.md).
- [Building custom `Estimator`s for a version of MNIST that uses CNNs](workshop_sections/mnist_series/mnist_cnn_custom_estimator/README.md), using either TensorFlow or [Keras](https://keras.io/) layers.


## 'Wide & Deep'

- [Using a tf.estimator to train a 'Wide & Deep' model](workshop_sections/wide_n_deep/README.md).

## Transfer Learning

- [Transfer learning: using a trained model to 'bootstrap' learning new classifications](workshop_sections/transfer_learning/README.md).
    + [Using Cloud ML](workshop_sections/transfer_learning/cloudml)
    + **(possibly outdated)** [Using a custom Estimator](workshop_sections/transfer_learning/TF_Estimator)

- **(possibly outdated)** [Building a word2vec model using a Custom Estimator, and exploring the learned embeddings](workshop_sections/word2vec/README.md). Introducing [TFRecords](https://www.tensorflow.org/api_guides/python/python_io).

## Extras

In addition, there is an [extras](workshop_sections/extras/README.md) directory, that contains some older labs not currently used in this workshop (and which may not necessarily run with the latest version of TF), but which may be of interest.
