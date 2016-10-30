
# Creating a Custom MNIST estimator

In a previous lab, we used `tf.contrib.tflearn.DNNClassifier` to easily build our MNIST model.
However, it turns out that a model that uses *convolutions* performs better on this task.  There is currently no "pre-baked" class that we can use (instead of `DNNClassifier`) for the CNN model.

So instead, we can build a *custom* [Estimator](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#Estimator).  We'll still need to define the model specifics, but we can leverage the Estimator's support for running training loops, model evaluations, and model predictions; as well as model checkpointing during training and generating info for TensorBoard.

This lab has two stages.  In the first stage, we'll see how we can take a pre-existing model graph definition and essentially plop it into an Estimator, making our new version much simpler.

In the second stage of the lab, we'll look at how we can use [layers](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#layers-contrib) to make our model specification more concise and easier to build and understand.

### A CNN version of MNIST using TensorFlow's "low-level" API

The [`mnist_cnn.py`](mnist_cnn.py) script follows the [Deep MNIST for Experts](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts) tutorial on the [tensorflow.org](http://tensorflow.org) site.

We can use this code, and its model, as a starting point for the Estimator(s) that we're going to build.

### MNIST-CNN using a Custom Estimator

The [`mnist_cnn_estimator.py`](mnist_cnn_estimator.py) script shows how you can create a custom [Estimator](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#Estimator) based on that same model graph, using TensorFlow's high-level `contrib.tflearn` API.

As with the "canned" Estimators, like the [DNNClassifier](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#DNNClassifier) used in one of the [other MNIST labs](../02_README_mnist_tflearn.md), the `Estimator`provides support for the `fit()`, `evaluate()`, and `predict()` functions so that you don't have to write a training loop, manage model checkpointing, etc., yourself.

We'll first walk through this version-- which essentially drops the original model graph into an `Estimator`-- then as an exercise, make the code even simpler by using *layers*.

## Exercise: convert the "low-level API" Estimator model to one that uses *layers*

Start with [`mnist_cnn_estimator.py`](mnist_cnn_estimator.py), which shows how you can wrap a model
graph in an `Estimator`.

Modify this script to replace the model definition code-- the code in the `model_fn` function-- with simpler code that defines the model graph using the [tf.contrib.layers](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#layers-contrib) library.

When you're done, try comparing the output of both training runs-- that of `mnist_cnn_estimator.py` and your new layers version-- in TensorBoard.


Note: The [transfer learning](../../transfer_learning/README.md) and [word2vec](../../word2vec/README.md) labs show additional examples of building and using custom Estimators.
