
# MNIST labs

This directory contains a series of labs/examples that use the ['MNIST'](http://yann.lecun.com/exdb/mnist/) (and ['fashion MNIST'](https://github.com/zalandoresearch/fashion-mnist)) databases.
Each builds conceptually on the previous ones.  Start at the README numbered '01', and work upwards.

- [01_README_mnist_simple](./01_README_mnist_simple.md): A simple version of MNIST with no hidden layers.

- [02_README_mnist_estimator](./02_README_mnist_estimator.md): Use the high-level TensorFlow APIs in `tf.estimator` to easily build a `LinearClassifier` and a `DNNClassifier` with hidden layers. Introducing TensorBoard.

**TBD**: Building a CNN Custom Estimator: both TensorFlow and [Keras](https://keras.io/) versions. Run locally or do (distributed) training on CMLE.
[mnist_cnn_custom_estimator](mnist_cnn_custom_estimator).
