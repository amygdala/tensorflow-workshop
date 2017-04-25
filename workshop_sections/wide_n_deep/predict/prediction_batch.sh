#!/bin/bash

export JOB_NAME=census_prediction_$(date +"%Y%m%d_%H%M%S")

gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model wnd1 \
    --version vCMD2 \
    --data-format TEXT \
    --region us-central1 \
    --input-paths gs://cloudml-engine/wide_n_deep/test_data/census.json \
    --output-path gs://cloudml-engine/wide_n_deep/predictions

