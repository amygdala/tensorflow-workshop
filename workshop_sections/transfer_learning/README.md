
This directory contains two examples of transfer learning using the "Inception V3" image classification model.

The [cloudml](cloudml) shows how to use Dataflow (Beam) to do image preprocessing, then train and serve your model on Cloud ML.  It supports distributed training on Cloud ML.
It is based on the example [here](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers), with some additional modifications to make it easy to use other image sets, and a prediction web server that demos how to use the Cloud ML API for prediction once your trained model is serving.

The [TF_Estimator](TF_Estimator) example uses TensorFlow too, but is not packaged to run on Cloud ML. It shows an example of using an `Estimator` class from the "tf.learn" library.
