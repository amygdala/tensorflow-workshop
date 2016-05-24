
# Installation instructions for the TensorFlow workshop

## Install Conda + Python 3 to use as your local virtual environment

Anaconda is a Python distribution that includes a large number of standard numeric and scientific computing packages. Anaconda uses a package manager called "conda" that has its own environment system similar to Virtualenv.

Install the version of Conda that **uses Python 3.5** by default.  Follow the instructions [here](https://www.continuum.io/downloads).  The [miniconda version](http://conda.pydata.org/miniconda.html) should suffice.

## Install TensorFlow into a Conda environment

Follow the instructions [on the TensorFlow site](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#anaconda-environment-installation) to create a Conda environment, *activate* it, and use pip to install TensorFlow within it.  When following these instructions, be sure to use the Python 3 variant for both environment creation and in grabbing the TensorFlow .whl file.

Remember to activate this environment in all the terminal windows you use during this workshop.

## Install some Python packages

With your conda environment activated, install the following packages:

```sh
$ conda install numpy
$ conda install scipy
$ pip install sklearn
$ conda install matplotlib
$ conda install jupyter
```


## Download data files for the workshop exercises

At various stages in this workshop, we'll have you download some data files. For convenience, we list them here:

https://storage.googleapis.com/oscon-tf-workshop-materials/saved_word2vec_model.zip
https://storage.googleapis.com/oscon-tf-workshop-materials/processed_reddit_data/reddit_post_title_words.zip
https://storage.googleapis.com/oscon-tf-workshop-materials/processed_reddit_data/news_aww/reddit_data.zip
https://storage.googleapis.com/oscon-tf-workshop-materials/learned_word_embeddings/reddit_embeds.zip

(Thanks to [reddit](https://www.reddit.com/), for allowing us to use some post data for a training corpus.)


## Optional: Clone/Download the TensorFlow repo from GitHub

We'll be looking at some examples based on code in the tensorflow repo. While it's not necessary, you might want to clone or download it [here](https://github.com/tensorflow/tensorflow), or grab the 0.8 release [here](https://github.com/tensorflow/tensorflow/releases).


## Optional: Download Kubernetes, and set up a Google Cloud Platform account as necessary

In one section of the workshop, we'll look at running the TensorFlow distributed runtime on a [Kubernetes](http://kubernetes.io/) cluster.
Kubernetes is Google's open-source container orchestration framework. It provides a useful framework to run microservice-based apps.

If you want to play along, create a Google Cloud Platform account and project ahead of time -- start with the 'try it free' button on [this page](https://cloud.google.com/) if you don't already have an account.
(Alternately, understand how to stand up a Kubernetes cluster on some other cloud provider).

Then, download and install the [latest Kubernetes release](https://github.com/kubernetes/kubernetes/releases).

