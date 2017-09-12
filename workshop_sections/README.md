
# TensorFlow workshop lab sections

This directory contains the workshop labs. [** add note re: TF versions tested with **].

## Getting Started

- [Building a small starter TensorFlow graph](getting_started/starter_tf_graph/README.md)
- [XOR: A minimal training example](getting_started/xor/README.md)

## The MNIST (& 'fashion MNIST') series

- [Introducing MNIST, and building a simple linear classifier in TensorFlow](mnist_series/01_README_mnist_simple.md).
- [Using TensorFlow's high-level APIs to build an MNIST DNN Classifier, and introducing TensorBoard](mnist_series/02_README_mnist_tflearn.md).
- [Building custom `Estimator`s for a version of MNIST that uses CNNs](mnist_series/mnist_cnn_custom_estimator/README.md), using either TensorFlow or [Keras](https://keras.io/) layers.


## 'Wide & Deep'

- [Using a tf.estimator to train a 'Wide & Deep' model](wide_n_deep/README.md).

## Transfer Learning

- [Transfer learning: using a trained model to 'bootstrap' learning new classifications](transfer_learning/README.md).
    + [Using Cloud ML](transfer_learning/cloudml)
    + **move this?** [Using a custom Estimator](transfer_learning/TF_Estimator)

- **move this?** [Building a word2vec model using a Custom Estimator, and exploring the learned embeddings](word2vec/README.md). Introducing [TFRecords](https://www.tensorflow.org/versions/r0.11/api_docs/python/python_io.html#data-io-python-functions).


## Extras

In addition, there is an [extras](extras/README.md) directory, that contains some older labs not currently used in this workshop (& which may not necessarily run with the latest version of TF), but which may be of interest.
