

This directory contains an example of transfer learning using the "Inception V3" image classification model.

**Note**: This example still works (and is fun to play with), but uses TensorFlow v1.2.  So it doesn't demonstrate current best practices.

The [cloudml](cloudml) example shows how to use [Cloud Dataflow](https://cloud.google.com/dataflow/) ([Apache
Beam](https://beam.apache.org/)) to do image preprocessing, then train and serve your model on Cloud ML.  It supports
distributed training on Cloud ML.
It is based on the example [here](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers), with
some additional modifications to make it easy to use other image sets, and a prediction web server that demos how to
use the Cloud ML API for prediction once your trained model is serving.

The list of image sources for the images used in the "hugs/no-hugs" training is here:
https://storage.googleapis.com/oscon-tf-workshop-materials/transfer_learning/hugs_photos_sources.csv
