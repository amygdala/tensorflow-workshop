

# Installation instructions for the TensorFlow workshop

  - [Docker-based installation](#docker-based-installation)
    - [Download the container image](#download-the-container-image)
    - [Create a directory to hold data files needed by the workshop](#create-a-directory-to-hold-data-files-needed-by-the-workshop)
    - [Run the container](#run-the-container)
    - [Restarting the container later](#restarting-the-container-later)
  - [Virtual environment-based installation](#virtual-environment-based-installation)
    - [Install Conda + Python 2.7 to use as your local virtual environment](#install-conda--python-27-to-use-as-your-local-virtual-environment)
    - [Install TensorFlow into a virtual environment](#install-tensorflow-into-a-virtual-environment)
    - [Install some Python packages](#install-some-python-packages)
    - [Install the Google Cloud SDK](#install-the-google-cloud-sdk)
    - [Cloud ML setup](#cloud-ml-setup)
    - [Cloud ML SDK installation (for 'transfer learning' preprocessing)](#cloud-ml-sdk-installation-for-transfer-learning-preprocessing)
  - [Set up some data files used in the examples](#set-up-some-data-files-used-in-the-examples)
  - [Optional: Clone/Download the TensorFlow repo from GitHub](#optional-clonedownload-the-tensorflow-repo-from-github)

You can set up for the workshop in two different, mutually-exclusive ways:

- [Running in a docker container](#docker-based-installation).
- [Installing the necessary packages into a virtual environment](#virtual-environment-based-installation).

## Docker-based installation

We're providing a [Docker](https://www.docker.com/) container image with all the necessary libraries included, for you to download.

To use it, you'll need to have [Docker installed](https://docs.docker.com/engine/installation/). To run some of the examples, you'll likely need to configure it with at least 4GB of memory.

### Download the container image

Once Docker is installed and running, download the workshop image:

```sh
$ docker pull gcr.io/google-samples/tf-workshop:v5
```

[Here's the Dockerfile](https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_image) used to build this image.

### Create a directory to hold data files needed by the workshop

Create a directory (called, say, `workshop-data`) to mount as a volume when you start up the docker container.  You can put data/output in this directory so that it is accessible both within and outside the running container.

### Run the container

Once you've downloaded the container image, you can run it like this:

```sh
$ docker run -v `pwd`/workshop-data:/root/tensorflow-workshop-master/workshop-data -it \
    -p 6006:6006 -p 8888:8888 gcr.io/google-samples/tf-workshop:v5
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

We highly recommend that you use a virtual environment for your TensorFlow installation rather than a direct install onto your machine.  The instructions below walk you thorough a `conda` install, but a `virtualenv` environment will work as well.

Note: The 'preprocessing' stage in the [Cloud ML transfer learning](workshop_sections/transfer_learning/cloudml)
example requires installation of the Cloud ML SDK, which requires Python 2.7. Otherwise, Python 3 should likely work.

### Install Conda + Python 2.7 to use as your local virtual environment

Anaconda is a Python distribution that includes a large number of standard numeric and scientific computing packages. Anaconda uses a package manager called "conda" that has its own environment system similar to Virtualenv.

Follow the instructions [here](https://www.continuum.io/downloads).  The [miniconda version](http://conda.pydata.org/miniconda.html) should suffice.

### Install TensorFlow into a virtual environment

Follow the instructions [on the TensorFlow site](https://www.tensorflow.org/get_started/os_setup#anaconda_installation) to create a Conda environment with Python 2.7, *activate* it, and then install TensorFlow within it.

**Note**: Install TensorFlow version 0.12.

If you'd prefer to use virtualenv, see [these instructions](https://www.tensorflow.org/get_started/os_setup#virtualenv_installation) instead.

Remember to activate your environment in all the terminal windows you use during this workshop.

### Install some Python packages

With your conda environment activated, install the following packages:

```sh
$ conda install numpy
$ conda install scipy
$ pip install sklearn pillow
$ conda install matplotlib
$ conda install jupyter
$ conda install nltk
```

(If you are using `virtualenv` instead of `conda`, install the packages using the equivalent `pip` commands instead).

Then run:
```
$ python -c "import nltk; nltk.download('punkt')"
```
(This will give you the necessary corpuses for the [word2vec](workshop_sections/word2vec/README.md) lab).

### Install the Google Cloud SDK

Follow the installation instructions [here](https://cloud.google.com/sdk/downloads) then run:

```
gcloud components install beta
```

To get the `gcloud beta ml` commands.

### Cloud ML setup

Follow the instructions below to create a project, initialize it for Cloud ML, and set up a storage bucket to use for the workshop examples.

* [Setting Up Your GCP Project](https://cloud.google.com/ml/docs/how-tos/getting-set-up#setting_up_your_google_cloud_project )
* [Initializing Cloud ML for your project](https://cloud.google.com/ml/docs/how-tos/getting-set-up#initializing_your_product_name_short_project)
* [Setting up your Cloud Storage Bucket](https://cloud.google.com/ml/docs/how-tos/getting-set-up#setting_up_your_cloud_storage_bucket)

### Cloud ML SDK installation (for 'transfer learning' preprocessing)

The Cloud ML SDK is needed to run the 'preprocessing' stage in the [Cloud ML transfer
learning](workshop_sections/transfer_learning/cloudml) example. It requires Python 2.7 to install. It's possible to
skip this part of setup for most of the exercises.

To install the SDK, follow the setup instructions
[on this page](https://cloud.google.com/ml/docs/how-tos/getting-set-up).
(Assuming you've followed the instructions above, you will have already done some of these steps. **Install TensorFlow version 0.12** as described in [this section](#install-tensorflow-into-a-virtual-environment), not 0.11)

You don't need to download the Cloud ML samples or docs for this workshop, though you may find it useful to grab them
anyway.

## Set up some data files used in the examples

### Transfer learning example

Because we have limited workshop time, we've saved a set of
[TFRecords]([TFRecords](https://www.tensorflow.org/api_docs/python/python_io/))
generated as part of the [Cloud ML transfer learning](workshop_sections/transfer_learning/cloudml) 
example. To save time, copy them now to your own bucket as follows.

Copy a zip of the generated records to some directory on your local machine:

```shell
gsutil cp gs://oscon-tf-workshop-materials/transfer_learning/cloudml/hugs_preproc_tfrecords.zip .
```

and then expand the zip:

```shell
unzip hugs_preproc_tfrecords.zip
```

Set the `BUCKET` variable to point to your GCS bucket (replacing `your-bucket-name` with the actual name):

```shell
BUCKET=gs://your-bucket-name
```

Then set the `GCS_PATH` variable as follows, and copy the unzipped records to a `preproc` directory under that path:

```shell
GCS_PATH=$BUCKET/hugs_preproc_tfrecords
gsutil cp -r hugs_preproc_tfrecords/ $GCS_PATH/preproc
```

Once you've done this, you can delete the local zip and `hugs_preproc_tfrecords` directory.

## Optional: Clone/Download the TensorFlow repo from GitHub

We'll be looking at some examples based on code in the tensorflow repo. While it's not necessary, you might want to clone or download it [here](https://github.com/tensorflow/tensorflow), or grab the latest release [here](https://github.com/tensorflow/tensorflow/releases).

