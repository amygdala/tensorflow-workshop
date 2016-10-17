
# Installation instructions for the TensorFlow workshop

You can set up for the workshop in two different, mutually-exclusive ways:

- [Running in a docker container](#docker-based-installation).
- [Installing the necessary packages into a virtual environment](#virtual-environment-based-installation).

## Docker-based installation

We're providing a [Docker](https://www.docker.com/) container image with all the necessary libraries included, for you to download.

To use it, you'll need to have [Docker installed](https://docs.docker.com/engine/installation/). To run some of the examples, you'll likely need to configure it with at least 4GB of memory.

### Download the container image

Once Docker is installed and running, download the workshop image:

```sh
$ docker pull gcr.io/google-samples/tf-workshop:v3
```

[Here's the Dockerfile](https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_image) used to build this image.

### Create a directory to hold data files needed by the workshop

Create a directory (called, say, `workshop-data`) to mount as a volume when you start up the docker container.  You can put data/output in this directory so that it is accessible both within and outside the running container.

### Run the container

Once you've downloaded the container image, you can run it like this:

```sh
$ docker run -v `pwd`/workshop-data:/root/tensorflow-workshop-master/workshop-data -it \
    -p 6006:6006 -p 8888:8888 gcr.io/google-samples/tf-workshop:v3
```

Edit the path to the directory you're mounting as appropriate. The first component of the `-v` arg is the local directory, and the second component is where you want to mount it in your running container.

### Restarting the container later

If you later exit your container and then want to restart it again, you can find the container ID by running:

```sh
$ docker ps -a
```

Then, run:

```sh
$ docker start <container_id>
```

(`docker ps` should then show it running). Once the workshop container is running again, you can exec back into it like this:

```sh
$ docker exec -it <container_id> bash
```

## Virtual environment-based installation

(These steps are not necessary if you have already completed the instructions for running the Docker image.)

We highly recommend that you use a virtual environment for your TensorFlow installation rather than a direct install onto your machine.  The instructions below walk you thorough a `conda` install, but a `virtualenv` environment should work as well. The instructions specify using Python 3, but Python 2.7 is also okay.

### Install Conda + Python 3 to use as your local virtual environment

Anaconda is a Python distribution that includes a large number of standard numeric and scientific computing packages. Anaconda uses a package manager called "conda" that has its own environment system similar to Virtualenv.

Install the version of Conda that uses Python 3.5 by default.  Follow the instructions [here](https://www.continuum.io/downloads).  The [miniconda version](http://conda.pydata.org/miniconda.html) should suffice.

### Install TensorFlow into a virtual environment

Follow the instructions [on the TensorFlow site](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#anaconda-installation) to create a Conda environment, *activate* it, and then use [conda-forge](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#using-conda) to install TensorFlow within it.

**Note**: as of this writing, `conda-forge` installs TensorFlow 0.10. That is fine for this workshop. If you'd prefer to install 0.11, follow the ["using pip" section](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#using-pip) instead.

If you'd prefer to use virtualenv, see [these instructions](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#virtualenv-installation) instead.

Remember to activate your environment in all the terminal windows you use during this workshop.

### Install some Python packages

With your conda environment activated, install the following packages:

```sh
$ conda install numpy
$ conda install scipy
$ pip install sklearn
$ conda install matplotlib
$ conda install jupyter
```

(If you are using `virtualenv` instead of `conda`, install the packages using the equivalent `pip` commands instead).

## Optional: Clone/Download the TensorFlow repo from GitHub

We'll be looking at some examples based on code in the tensorflow repo. While it's not necessary, you might want to clone or download it [here](https://github.com/tensorflow/tensorflow), or grab the latest release [here](https://github.com/tensorflow/tensorflow/releases).

