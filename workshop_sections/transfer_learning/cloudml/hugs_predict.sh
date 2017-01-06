#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up

# Now that we are set up, we can start processing some 'hugs/no-hugs' images.

if [ -z "$1" ]
  then
    echo "No GCS_PATH supplied"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No VERSION_NAME supplied"
    exit 1
fi


# declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
# declare -r JOB_ID="hugs_${USER}_$(date +%Y%m%d_%H%M%S)"
# declare -r BUCKET="gs://aju-vtests2-ml-amy"
# declare -r JOB_ID=$1
declare -r GCS_PATH=$1
declare -r DICT_FILE=gs://oscon-tf-workshop-materials/transfer_learning/cloudml/hugs_photos/dict.txt

declare -r MODEL_NAME=hugs
declare -r VERSION_NAME=$2

echo
echo "Using job id: " $JOB_ID
echo "Using GCS_PATH: " $GCS_PATH
echo "Using VERSION_NAME: " $VERSION_NAME
set -v -e


# Tell CloudML about a new type of model coming.  Think of a "model" here as
# a namespace for deployed Tensorflow graphs.
gcloud beta ml models create "$MODEL_NAME"

# Each unique Tensorflow graph--with all the information it needs to execute--
# corresponds to a "version".  Creating a version actually deploys our
# Tensorflow graph to a Cloud instance, and gets is ready to serve (predict).
gcloud beta ml versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model"

# # Models do not need a default version, but its a great way move your production
# # service from one version to another with a single gcloud command.
# gcloud beta ml versions set-default "$VERSION_NAME" --model "$MODEL_NAME"

# # Finally, download a daisy and so we can test online prediction.
# gsutil cp \
#   gs://cloud-ml-data/img/flower_photos/daisy/100080576_f52e8ee070_n.jpg \
#   daisy.jpg

# # Since the image is passed via JSON, we have to encode the JPEG string first.
# python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' daisy.jpg &> request.json

# # Here we are showing off CloudML online prediction which is still in alpha.
# # If the first call returns an error please give it another try; likely the
# # first worker is still spinning up.  After deploying our model we give the
# # service a moment to catch up--only needed when you deploy a new version.
# # We wait for 10 minutes here, but often see the service start up sooner.
# sleep 10m
# gcloud beta ml predict --model ${MODEL_NAME} --json-instances request.json
