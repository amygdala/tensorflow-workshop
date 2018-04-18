
# MNIST labs

This directory contains a series of labs/examples that introduce the the high-level TensorFlow APIs in [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator) and [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) via the ['MNIST'](http://yann.lecun.com/exdb/mnist/) (and ['fashion MNIST'](https://github.com/zalandoresearch/fashion-mnist)) data sets.

- [01_README_mnist_simple](./01_README_mnist_simple.md): A simple version of a linear classifier for MNIST that uses the low-level TensorFlow APIs.

   Run this example as a [colab notebook](https://colab.sandbox.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/mnist_series/mnist_simple.ipynb).

- [mnist_estimator.ipynb](./mnist_estimator.ipynb): Introducing the high-level TensorFlow APIs in [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator) and [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) to easily build a `LinearClassifier`, as well a `DNNClassifier` with hidden layers. Introducing [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

   Run this example as a [colab notebook](https://colab.sandbox.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/mnist_series/mnist_estimator.ipynb)

- [Building Custom CNN Estimators](mnist_cnn_custom_estimator): Where 'canned' Estimators aren't available for your use case, you can build a custom one, to get all the advantages of using an Estimator, including support for distributed training. You can use Keras layers to do this. Examples show how to do this for variants of CNNs, with both Keras and TensorFlow layers.  
Click through to the [README](mnist_cnn_custom_estimator/README.md) for links to run these examples as colab notebooks.
