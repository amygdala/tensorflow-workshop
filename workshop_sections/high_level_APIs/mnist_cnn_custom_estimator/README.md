
# Creating and using a Custom MNIST Estimator

We saw earlier in the workshop that a 'canned' DNNClassifier worked pretty well with 'regular'
MNIST, but did not do nearly as well with
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

Let's see if we can improve that using a convolutional model.  There is not currently a canned
Estimator suitable for our purposes, so we'll build a custom model.

A good way to do this is to specify your model logic  via either or both of `tf.layers` and
`keras.layers`; then create a custom
[Estimator](https://www.tensorflow.org/programmers_guide/estimators) that uses that model. The
Estimator class gives some useful benefits, including checkpointing, TensorBoard integrations, and
`tensorflow/serving` export.  In addition, **Estimators make it easy to do distributed training**:
you can run Estimators-based models on a local host or on a distributed multi-server environment
without changing your model, and can run Estimators-based models on CPUs, GPUs, or TPUs without
recoding your model.

The nice thing about the interchangeability of TF and Keras layers is that if we have a Keras model,
we can essentially drop it in to "Estimator boilerplate" to get the
Estimator's [distribution benefits](https://www.tensorflow.org/deploy/distributed).
[Note: in TF 1.4, the syntax for creating an Estimator from a Keras model will become simpler than
shown in the current code.]

## Building a Custom Estimator

When you're building a custom Estimator, you need to specify a `model_fn` that describes your model
logic, in addition to the input function(s) that specify how your data is fed into the model.

We can Keras layers to define the `model_fn`; or interchangeably, we can use `tf.layers`.

### A Custom Estimator CNN that uses keras.layers

Here's a custom Estimator that implements a convolutional neural net for MNIST data, using Keras layers: [cnn_mnist_keras.ipynb](cnn_mnist_keras.ipynb).

[Open this notebook in colab](https://colab.research.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/cnn_mnist_keras.ipynb).

### A Custom Estimator CNN that uses tf.layers

We've also built a version that uses Keras layers instead of TF layers.
The notebook is here: [cnn_mnist_tf.ipynb](cnn_mnist_tf.ipynb).

The custom Estimator 'boilerplate' remains the same, but this time TensorFlow layers are used for
constructing the inference model.

[Open this notebook in colab](https://colab.research.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/cnn_mnist_tf.ipynb).

You can see that these two models give similar results when trained on 'regular' MNIST data:

<!-- <a href="https://storage.googleapis.com/amy-jo/images/tf-workshop/cnn_regmnist.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/tf-workshop/cnn_regmnist.png" width="600"/></a> -->

<a href="https://storage.googleapis.com/amy-jo/images/tf-workshop/tf_keras_regmnist.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/tf-workshop/tf_keras_regmnist.png" width="600"/></a>

## Now let's raise the stakes... *Fashion-MNIST*

What happens if we use the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset instead of 'regular' MNIST? Our CNN models no longer do so well (though — they're still doing better than the [DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) did on this dataset).  Note the difference in evaluation loss as compared to 'regular' MNIST.

<a href="https://storage.googleapis.com/amy-jo/images/tf-workshop/fashion_vs_reg_cnn.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/tf-workshop/fashion_vs_reg_cnn.png" width="600"/></a>

### Hmm, let's try another model

If you do a web search, you can see all sorts of Fashion-MNIST benchmarking against different models.

The one we'll play with here is based on
[this blog post](http://www.sas-programming.com/2017/09/a-vgg-like-cnn-for-fashion-mnist-with.html) --
a [VGG-like]( http://www.robots.ox.ac.uk/~vgg) CNN (VGG is a deep convolutional network for object
recognition developed and trained by Oxford's Visual Geometry Group, which achieved
good performance on the ImageNet dataset.)

Our custom Estimator implementation is here:
[cnn_mnist_keras_vgg_like.ipynb](cnn_mnist_keras_vgg_like.ipynb).  To create it, we replaced our
original Keras layers with the 'VGG-like' Keras layers.  The custom Estimator lets us do easy
distributed training of this model.

[Open this notebook in colab](https://colab.research.google.com/github/amygdala/tensorflow-workshop/blob/master/workshop_sections/high_level_APIs/mnist_cnn_custom_estimator/cnn_mnist_keras_vgg_like.ipynb).

We can see from the comparison graph below that the VGG-like model gets better prediction accuracy even though its loss values are not so low (our original CNN models look to be overfitting on this dataset).

<a href="https://storage.googleapis.com/amy-jo/images/tf-workshop/vgg_vs_reg_cnn.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/tf-workshop/vgg_vs_reg_cnn.png" width="600"/></a>

So this new model gives a bit of an improvement!

But with more complex models, and the harder dataset, it's getting harder to train locally in a
reasonable time period (at least on CPUs).

## Optional: Distributed training with Estimator models: using Cloud ML Engine

[Note: This section and the GCP setup is not an official part of this workshop, though you might
[want to play around with this stuff later on your own.]

Distributed training lets us get results faster, and we need to train for more steps than we have
been to see whether the VGG-like CNN gives improved results.

The [trainer](trainer) directory shows how you'd set things up to do distributed training of your
models. 

To run the examples below, first
[set up your environment for working with Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)
(CMLE).
[trainer.task](trainer/task.py) and [trainer.keras_task](trainer/keras_task.py) use our
'original' CNN model, and [trainer.keras_vgg_task](trainer/keras_vgg_task.py) uses our new more
complex CNN.

### Training locally

Once the `gcloud` SDK is installed, you can run training locally via the SDK as follows.
This can be useful for debugging.


```
gcloud ml-engine local train --package-path trainer --module-name trainer.keras_task \
  -- --num_steps=10000
```

You can point the `--data_dir` flag to your downloaded Fashion-MNIST directory to train on that dataset.

### Training on CMLE (& Using Fashion-MNIST)

To train on Cloud ML Engine, use command syntax like the following instead. (While not shown here,
you can modify the `scale-tier` arg, or pass a config file, to run on different cluster
configurations, including use of GPUs.
See the [documentation](https://cloud.google.com/ml-engine/docs/) for more info.)

To use Fashion-MNIST for the dataset,
point `--data_dir` to a GCS directory containing the *unzipped* Fashion-MNIST dataset files, as indicated below.
If you omit the `--data_dir` argument, the training will use 'regular' MNIST.

This trains the Keras-layer-based version of the first CNN model.

```
gcloud ml-engine jobs submit training cnn_mnist_$(date -u +%y%m%d_%H%M%S) \
  --job-dir gs://<your-gcs-bucket-name>/cnn_tests_$(date -u +%y%m%d_%H%M%S) --scale-tier STANDARD_1 \
  --runtime-version 1.6 \
  --module-name trainer.keras_task \
  --package-path trainer --config config_custom_gpus.yaml\
  --region us-east1 -- --num_steps 65000  --data_dir=gs://<your-gcs-bucket-name>/path/to/fashion_mnist/ --num_steps=1000
```

