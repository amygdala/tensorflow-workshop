
This directory contains versions of scripts in the parent directory that are packaged to facilitate distributed training on CMLE. See more info in the [parent README](../README.md).
The `../setup.py` file contains info about how to package this module for deployment.

The `mnist_input.py` file is a copy of `tensorflow/contrib/learn/python/learn/datasets/mnist.py`, altered to allow passing in a different source URL.  This makes it straightforward to use either the MNIST or fashion-MNIST dataset.
