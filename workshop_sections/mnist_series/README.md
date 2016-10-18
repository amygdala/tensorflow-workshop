
# MNIST labs

This directory contains a series of labs/examples using the ['MNIST'](http://yann.lecun.com/exdb/mnist/) database.
Each builds conceptually on the previous ones.  Start at the README numbered '01', and work upwards.

- [01_README_mnist_simple](./01_README_mnist_simple.md): A simple version of MNIST with no hidden layers.

- [02_README_mnist_tflearn](./02_README_mnist_tflearn.md): Use the high-level TensorFlow API in `contrib.tflearn` to easily build DNNs with hidden layers. Introducing TensorBoard.

- [03_README_mnist_layers](./03_README_mnist_layers.md): Build an MNIST DNN with hidden layers using the 'low-level' TensorFlow API. Saving and loading model checkpoints, and generating TensorBoard summaries.

- [04_README_mnist_estimator](./04_README_mnist_estimator): Bonus lab: Build a custom `Estimator` for a version of MNIST that uses CNNs.  We won't cover this in the workshop.
