
# Installation instructions for the TensorFlow workshop

## Install Conda + Python 3 to use as your local virtual environment

Anaconda is a Python distribution that includes a large number of standard numeric and scientific computing packages. Anaconda uses a package manager called "conda" that has its own environment system similar to Virtualenv.

Install the version of Conda that **uses Python 3.5** by default.  Follow the instructions [here](https://www.continuum.io/downloads).

## Install TensorFlow into a Conda environment

Follow the instructions [on the TensorFlow site](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#anaconda-environment-installation) to create a Conda environment, *activate* it, and use pip to install TensorFlow within it.  When following these instructions, be sure to use the Python 3 variant for both environment creation and in grabbing the TensorFlow .whl file.

Remember to activate this environment in all the terminal windows you use during this workshop.

## Install some Python packages

With your conda environment activated, intall the following packages:

```sh
$ conda install numpy
$ conda install sklearn
$ conda install matplotlib
$ conda install IPython
```

[** probably nicer way to do this and still use conda install instead of pip.  Also - is anything missing? **]

## Clone/Download the TensorFlow repo from GitHub

We'll be looking at some examples in the tensorflow repo. Clone or download it [here](https://github.com/tensorflow/tensorflow), or grab the 0.8 release [here](https://github.com/tensorflow/tensorflow/releases).  

## Download data files for the workshop exercises

At various stages in this workshop, we'll need to download some data files. For convenience, we list them all here.

[** TBD **]

