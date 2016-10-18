
# Creating a Custom MNIST estimator

This lab is not used in the workshop, but may be of interest.

The [`mnist_cnn.py`](mnist_cnn.py) script follows the [Deep MNIST for Experts](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts) tutorial on the tensorflow.org site.
(Its hidden layers use *convolutions*.)

The [`mnist_cnn_estimator.py`](mnist_cnn_estimator.py) script shows how you can create a custom [Estimator](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#Estimator) based on that same model graph, using TensorFlow's high-level `contrib.tflearn` API. 

As with the "canned" Estimators, like the [DNNClassifier](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#DNNClassifier) used one of the [other MNIST labs](02_README_mnist_tflearn.md), the Estimator provides support for the `fit()`, `eval()`, etc. functions so that you don't have to write a training loop, manage model checkpointing, etc., yourself.

See the [word2vec](../word2vec/README.md) lab for more on building custom Estimators.

