
# Installation

We suggest installing the necessary libraries locally on your laptop as described below in "__Local Installation__", as that will let you run TensorBoard as part of the workshop, and let you run python scripts as well as notebooks. 

But if you have trouble with your installation, you will be able to run everything but TensorBoard in [Colab notebooks instead](https://colab.research.google.com/).

## Local Installation

It's highly recommended to use a Python virtual environment for this workshop, either [Virtualenv](https://virtualenv.pypa.io/en/stable) or [Conda](http://conda.pydata.org/miniconda.html).  Either Python 2.7 or Python 3 will work.

Then install the following into your virtual environment:

-  Install [TensorFlow](https://www.tensorflow.org/install). Your TensorFlow version must be >=1.7 to run all the labs. Note that even if you're running a conda virtual environment, use `pip install tensorflow` to install TensorFlow (*not* the `conda install` version, which may not be up to date.)

-  Install [Jupyter](https://jupyter.org/install.html). 

- Optionally, install `matplotlib`. (If you're using a conda virtual environment, use `conda install matplotlib`.)

### Test your installation

Start up a jupyter notebooks server from the command line:

```sh
jupyter notebook .
```

Create a new notebook and paste the following python code into a notebook cell:

```python
import tensorflow as tf

print(tf.__version__)
```

'Run' the cell. Ensure that there are no import errors and that the TensorFlow version is as expected.

## Using Colaboratory

Instead of running in a local installation, all of the jupyter notebooks in this workshop can also be run on [Colab](https://colab.research.google.com/), with the exception of the lab sections that use TensorBoard. 

The lab section READMEs include links to launch the workshop notebooks in colab.


## Google Cloud Platform

Some of the labs include instructions for optionally running training and prediction on Google Cloud Platform (GCP) [Cloud ML Engine](https://cloud.google.com/ml-engine).  

If you want to play along, set up a GCP account: click the **Try it free** button at the top of this page:
[cloud.google.com](https://cloud.google.com/).


## Docker image

A Docker image is also available.

### Download the container image

Once Docker is installed and running, download the workshop image:

```sh
docker pull gcr.io/google-samples/tf-workshop:v8
```

[Here's the Dockerfile](https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_image) used to build this image.

### Run the container

Once you've downloaded the container image, run it like this:

```sh
docker run -it \
    -p 6006:6006 -p 8888:8888 -p 5000:5000 gcr.io/google-samples/tf-workshop:v8
```


