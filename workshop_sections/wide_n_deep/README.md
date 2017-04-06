# Wide & Deep with TensorFlow

This directory contains the code for running a Wide and Deep model. It also runs in Cloud ML Engine, using TensorFlow 1.0. This code has been tested on Python 2.7 and 3.5

# About the dataset and model
Wide and deep jointly trains wide linear models and deep neural networks -- to combine the benefits of memorization and generalization for recommender systems. See the [research paper](https://arxiv.org/abs/1606.07792) for more details. The code is based on the [TensorFlow wide and deep tutorial](https://www.tensorflow.org/tutorials/wide_and_deep/).

We will use the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) to predict the probability that the individual has an annual income of over 50,000 dollars. This data was extracted from the 1994 US Census by Barry Becker. 

The prediction task is to determine whether a person makes over 50K a year.

The dataset is downloaded as part of the script (and in the cloud directly uses the copy stored online).

# Training and evaluation
This repo presents 3 methods of running the model: locally, on a jupyter notebook, and on Google Cloud ML Engine.

The commands below assume you are in this directory (wide_n_deep). 

You should move to it with `cd workshop_sections/wide_n_deep`

### Local
`python widendeep/model.py`

### Jupyter Notebook
Run the notebook, and step through the cells.

`jupyter notebook`

### Google Cloud Machine Learning Engine
The workflow to run this on Cloud Machine Learning Engine is to do a local run first, then move to the cloud.

First, let's go to our working directory: 
`cd workshop_sections/wide_n_deep`

#### Test it locally:
    $ gcloud ml-engine local train --package-path=widendeep --module-name=widendeep.model
    TensorFlow version 1.0.0
    model directory = models/model_WIDE_AND_DEEP_1491431579
    estimator built
    fit done
    evaluate done
    Accuracy: 0.84125
    Model exported to models/model_WIDE_AND_DEEP_1491431579/exports

#### Run it in the cloud:
Ensure you have the project you want to work in selected. You can check using `gcloud config list`
    
If you need to set it to a new value, do so using `gcloud config set <YOUR_PROJECT_ID_HERE>`

You should also make sure that the Cloud ML Engine API is turned on for your project. More info about getting setup is here: https://cloud.google.com/ml-engine/docs/quickstarts/command-line

Next, set the following environment variables and submit a training job.

    gcloud config set compute/region us-central1
    gcloud config set compute/zone us-central1-c

    export PROJECT_ID=`gcloud config list project --format "value(core.project)"`
    export BUCKET=gs://${PROJECT_ID}-ml
    export JOB_NAME=widendeep_${USER}_$(date +%Y%m%d_%H%M%S)
    export TRAIN_PATH=${BUCKET}/${JOB_NAME}

    gcloud ml-engine jobs submit training ${JOB_NAME} --package-path=widendeep --module-name=widendeep.model  --region=us-central1 --job-dir=${TRAIN_PATH} 
    
    gcloud ml-engine jobs submit training ${JOB_NAME} --package-path=widendeep --module-name=widendeep.model --job-dir=${TRAIN_PATH} --config config.yaml
    
You can check the status of your training job with the command:

    gcloud ml-engine jobs describe $JOB_NAME
    
You can also see it's progress in your cloud console and view the logs.

To run another job (in your dev workflow), simply set a new `JOB_NAME` and `TRAIN_PATH` and then re-run the `jobs.summit` call. Job names must be unique.

    export JOB_NAME=widendeep_${USER}_$(date +%Y%m%d_%H%M%S)
    export TRAIN_PATH=${BUCKET}/${JOB_NAME}
    gcloud ml-engine jobs submit training ${JOB_NAME} --package-path=widendeep --module-name=widendeep.model  --region=us-central1 --job-dir=${TRAIN_PATH}

    
# Your trained model
Whether you ran your training locally or in the cloud, you should now have a set of files exported. If you ran this locally, it will be located in someplace similar to `models/model_WIDE_AND_DEEP_1234567890/exports/1234567890`. If you ran it in the cloud, it will be located in the GCS bucket that you passed.

The trained model files that were exported are ready to be used for prediction. 

# Prediction
You can run prediction jobs in Cloud ML Engine as well, using the Prediction Service. 

Before we begin, if you trained a model locally, you should upload the contents that were exported (something like `saved_model.pb` and a folder called `variables`) to a Google Cloud Storage location and make a note of its address (`gs://<BUCKET_ID>/path/to/model`)

Now we are ready to create a model
    
    export MODEL_NAME='my_model'
    gcloud ml-engine models create $MODEL_NAME

Next, create a 'version' of that model

    export VERSION_NAME='my_version'
    export DEPLOYMENT_SOURCE='gs://LOCATION_OF_MODEL_FILES'
    gcloud ml-engine versions create $VERSION_NAME --model $MODEL_NAME --origin $DEPLOYMENT_SOURCE
    
Finally, make a prediction with your newly deployed version!

    gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances test_instance.json
