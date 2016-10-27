
# Creating a Custom MNIST estimator

The [`mnist_cnn.py`](mnist_cnn.py) script follows the [Deep MNIST for Experts](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts) tutorial on the tensorflow.org site.
(Its hidden layers use *convolutions*.)

The [`mnist_cnn_estimator.py`](mnist_cnn_estimator.py) script shows how you can create a custom [Estimator](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#Estimator) based on that same model graph, using TensorFlow's high-level `contrib.tflearn` API. 

As with the "canned" Estimators, like the [DNNClassifier](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#DNNClassifier) used in one of the [other MNIST labs](../02_README_mnist_tflearn.md), the `Estimator`provides support for the `fit()`, `eval()`, etc. functions so that you don't have to write a training loop, manage model checkpointing, etc., yourself.

The [transfer learning](../../transfer_learning/README.md) and [word2vec](../../word2vec/README.md) labs will show additional examples of building and using custom Estimators.

## Exercise

Start with [`mnist_cnn_estimator.py`](mnist_cnn_estimator.py), which shows how you can wrap a model
graph in an `Estimator`.

Modify this script to replace the model definition code-- the code in the `model_fn` function-- with simpler code that defines the model graph using the [tf.contrib.layers](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#layers-contrib) library.
