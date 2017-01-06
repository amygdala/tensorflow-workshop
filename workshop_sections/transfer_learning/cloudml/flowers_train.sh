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

# TODO: usage

if [ -z "$1" ]
  then
    echo "No bucket supplied."
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No job id supplied"
    exit 1
fi

declare -r BUCKET=$1
declare -r JOB_ID=$2

if [ -z "$3" ]
  then
    declare -r GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
    echo "Auto-generating GCS path: ${GCS_PATH}"
  else
    declare -r GCS_PATH=$3
fi


echo
echo "Using job id: " $JOB_ID
echo "Using GCS_PATH: " $GCS_PATH
set -v -e


# Train on cloud ML.
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config.yaml \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*" \


# You can also separately run:
# gcloud beta ml jobs stream-logs "$JOB_ID"
# to see logs for a given job.

set +v

# You can also run the training locally via this command:
# TODO -- xxx

echo "Using GCS_PATH: " $GCS_PATH
