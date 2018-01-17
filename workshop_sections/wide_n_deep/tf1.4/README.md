
# Easy distributed training with TensorFlow using `tf.estimator.train_and_evaluate`

## Introduction

TensorFlow release 1.4 introduced the function [**`tf.estimator.train_and_evaluate`**](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate), which simplifies training, evaluation, and exporting of [`Estimator`](https://www.tensorflow.org/get_started/estimator) models. It abstracts away the details of [distributed execution](https://www.google.com/url?q=https://www.tensorflow.org/deploy/distributed) for training and evaluation, while also supporting local execution, and provides consistent behavior across both local/non-distributed and distributed configurations.

This means that using **`tf.estimator.train_and_evaluate`**, you can run the same code on both locally and distributed in the cloud, on different devices and using different cluster configurations, and get consistent results, **without making any code changes**. A train-and-evaluate loop is automatically supported. When you're done training (or at intermediate stages), the trained model is automatically exported in a form suitable for serving (e.g. for [Cloud ML Engine online prediction](https://cloud.google.com/ml-engine/docs/prediction-overview), or [TensorFlow serving](https://www.tensorflow.org/serving/)).

In this example, we'll walk through how to use `tf.estimator.train_and_evaluate` with an Estimator model, and then show how easy it is to do **distributed training of the model on [Cloud ML Engine](https://cloud.google.com/ml-engine) (CMLE)**, and to move between different cluster configurations with just a config tweak.
(The TensorFlow code itself supports distribution on any infrastructure (GCE, GKE, etc.) when properly configured, but we will focus on CMLE, which makes the experience seamless).

The primary steps necessary to do this, in addition to building your Estimator model, are to define how data is fed into the model for both training and test datasets (often these definitions are essentially the same), and to define training and eval specifications ([`TrainSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec) and [`EvalSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec)) passed to `tf.estimator.train_and_evaluate`.  The `EvalSpec` can include information on how to export your trained model for prediction (serving), and we'll look at how to do that as well.

Then we'll look at how to **use your trained model to make predictions**. 

The example also includes the use of [**Datasets**](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to manage our input data. This API is part of TensorFlow 1.4, and is an easier and more performant way to create input pipelines to TensorFlow models. (See [this article](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/performance/datasets_performance.md) for more on why input pipelining is so important, particularly when using accelerators).

For our example, we'll use the The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have hosted the data
on [Google Cloud Storage](https://cloud.google.com/storage/) (GCS) in a slightly cleaned form. We'll use this dataset to predict income category based on various information about a person.

This README omits some of the details of the example.
To see the specifics and work through the code yourself, visit the [Jupyter](http://jupyter.org/) notebook [in this directory](using_tf.estimator.train_and_evaluate.ipynb).
(The example in the [notebook](using_tf.estimator.train_and_evaluate.ipynb) is a slightly modified version of [this other example](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/estimator/trainer)).


## First step: create an Estimator

We'll first create an [`Estimator`](https://www.tensorflow.org/get_started/estimator) model using a prebuilt Estimator subclass, [`DNNLinearCombinedClassifier`](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier).
This is a "wide and deep" model.
Wide and deep models use a deep neural net (DNN) to learn high level abstractions about complex features or interactions between such features. These models then combine the outputs from the DNN with a [linear regression](https://en.wikipedia.org/wiki/Linear_regression) performed on simpler features. This provides a balance between power and speed that is effective on many structured data problems.
You can read more about this model and its use [here](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html). 

We're using Estimators because they give us built-in support for distributed training and evaluation (along with other nice features). You should nearly always use Estimators to create your TensorFlow models. You can build a Custom Estimator if none of the pre-made Estimators suit your purpose.

See the accompanying [notebook](https://nbviewer.jupyter.org/github/amygdala/code-snippets/blob/master/ml/census_train_and_eval/using_tf.estimator.train_and_evaluate.ipynb#First-step:-create-an-Estimator) for the details of defining our Estimator, including specifying the expected format of the input data.
The data is in csv format, and looks like this:

```
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
...
```

We'll use the last field, which indicates income bracket, as our label, meaning that this is the value we'll predict based on the values of the other fields.

In the [notebook](using_tf.estimator.train_and_evaluate.ipynb), we define a `build_estimator` function, which takes as input config info, and returns a `tf.estimator.DNNLinearCombinedClassifier` object.
We'll call it like this:

```python
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(model_dir=output_dir)

FIRST_LAYER_SIZE = 100  # Number of nodes in the first layer of the DNN
NUM_LAYERS = 4  # Number of layers in the DNN
SCALE_FACTOR = 0.7  # How quickly should the size of the layers in the DNN decay
EMBEDDING_SIZE = 8  # Number of embedding dimensions for categorical columns

estimator = build_estimator(
    embedding_size=EMBEDDING_SIZE,
    # Construct layers sizes with exponential decay
    hidden_units=[
        max(2, int(FIRST_LAYER_SIZE *
                   SCALE_FACTOR**i))
        for i in range(NUM_LAYERS)
    ],
    config=run_config
)
```


## Define input functions using Datasets

Now that we have defined our model structure, the next step is to use it for training and evaluation.
As with any `Estimator`, we'll need to tell the `DNNLinearCombinedClassifier` object how to get its training and eval data. We'll define a function (`input_fn`) that knows how to generate features and labels for training or evaluation, then use that definition to create the actual train and eval input functions.

We'll use [Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to access our data. 
This API is a new way to create [input pipelines to TensorFlow models](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/performance/datasets_performance.md). 
The `Dataset` API is much more performant than using `feed_dict` or the queue-based pipelines, and it's [cleaner and easier](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html) to use.

In this simple example, our datasets are too small for the use of the Datasets API to make a large difference, but with larger datasets it becomes much more important.

The `input_fn` definition is the following. It uses a couple of helper functions that are defined in the [notebook](https://nbviewer.jupyter.org/github/amygdala/code-snippets/blob/master/ml/census_train_and_eval/using_tf.estimator.train_and_evaluate.ipynb#Define-input-functions-(using-Datasets)).    
`parse_label_column` is used to convert the label strings (in our case, ' <=50K' and ' >50K') into [one-hot](https://en.wikipedia.org/wiki/One-hot) encodings.


```python
# This function returns a (features, indices) tuple, where features is a dictionary of
# Tensors, and indices is a single Tensor of label indices.
def input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=0,
                      batch_size=200):

  dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(parse_csv)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features, parse_label_column(features.pop(LABEL_COLUMN))
```

Then, we'll use `input_fn` to define both the `train_input` and `eval_input` functions.  We just need to pass `input_fn` the different source files to use for training versus evaluation.
As we'll see below, these two functions will be used to define a `TrainSpec` and `EvalSpec` used by `train_and_evaluate`.


```python
train_input = lambda: input_fn(
    TRAIN_FILES,
    batch_size=40
)

# Don't shuffle evaluation data
eval_input = lambda: input_fn(
    EVAL_FILES,
    batch_size=40,
    shuffle=False
)
```

## Define training and eval specs

Now we're nearly set.  We just need to define the the `TrainSpec` and `EvalSpec` used by `tf.estimator.train_and_evaluate`. These specify not only the input functions, but how to export our trained model.

First, we'll define the [`TrainSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec), which takes as an arg `train_input`:


```python
train_spec = tf.estimator.TrainSpec(train_input,
                                  max_steps=1000
                                  )
```

For our [`EvalSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec), we'll instantiate it with something additional -- a list of exporters, that specify how to export a trained model so that it can be used for serving.

To specify our exporter, we first define a *serving input function*.  A serving input function should produce a [ServingInputReceiver](https://www.tensorflow.org/api_docs/python/tf/estimator/export/ServingInputReceiver).

A `ServingInputReceiver` is instantiated with two arguments — `features`, and `receiver_tensors`. The `features` represent the inputs to our Estimator when it is being served for prediction. The `receiver_tensor` represent inputs to the server.  

These two arguments will not necessarily always be the same — in some cases we may want to perform some tranformation(s) before feeding the data to the model. [Here's](https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer/model.py#L197) one example of that, where the inputs to the server (csv-formatted rows) include a field to be removed.

However, in our case, the inputs to the server are the same as the features input to the model. 


```python
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
```

Then, we define an [Exporter](https://www.tensorflow.org/api_docs/python/tf/estimator/FinalExporter) in terms of that serving input function, and pass the `EvalSpec` constructor a list of exporters.
(We're just using one exporter here, but if you define multiple exporters, training will result in multiple saved models).


```python
exporter = tf.estimator.FinalExporter('census',
      json_serving_input_fn)
eval_spec = tf.estimator.EvalSpec(eval_input,
                                steps=100,
                                exporters=[exporter],
                                name='census-eval'
                                )
```

## Train your model using `train_and_evaluate`


Now, we have defined everything we need to train and evaluate our model, and export the trained model for serving, via a call to **`train_and_evaluate`**:


```python
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

This call will train your model and exported the result in a format that makes it easy to use it for prediction! 

Using `train_and_evaluate`, the training behavior will be consistent whether you run this function in a local/non-distributed context or in a distributed configuration.

The exported trained model can be served on many platforms. You may particularly want to consider ways to scalably serve your model, in order to handle many prediction requests at once— say if you're using your model in an app you're building, and you expect it to become popular.  [Cloud ML Engine online prediction](https://cloud.google.com/ml-engine/docs/prediction-overview), or [TensorFlow serving](https://www.tensorflow.org/serving/)) are two of the options for doing this.

In this example, we'll look at using **CMLE Online Prediction**. But first, let's take a closer look at our exported model.

### Examine the signature of the exported model.

TensorFlow ships with a CLI that allows you to inspect the *signature* of exported binary files. This can be useful as a sanity check. 
It's run as follows, by passing it the path to directory containing the saved model, which will be called `saved_model.pb`.
For our model, it will be found under `$output_dir/export/census`.  This is because we passed the `census` name to our `FinalExporter` above.  (`$output_dir` was specified when we constructed our estimator). 

```sh
saved_model_cli show --dir $output_dir/export/census/<timestamp> --tag serve --signature_def predict
```

The `saved_model_cli` command shows us this info (abbreviated for conciseness):

```
The given SavedModel SignatureDef contains the following input(s):
inputs['age'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1)
    name: Placeholder_8:0
inputs['capital_gain'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1)
    name: Placeholder_10:0
inputs['capital_loss'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1)
    name: Placeholder_11:0
inputs['education'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: Placeholder_2:0
inputs['education_num'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1)
    name: Placeholder_9:0
<... more input fields here ...>
The given SavedModel SignatureDef contains the following output(s):
outputs['class_ids'] tensor_info:
    dtype: DT_INT64
    shape: (-1, 1)
    name: head/predictions/classes:0
outputs['classes'] tensor_info:
    dtype: DT_STRING
    shape: (-1, 1)
    name: head/predictions/str_classes:0
outputs['logistic'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: head/predictions/logistic:0
outputs['logits'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: head/predictions/logits:0
outputs['probabilities'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 2)
    name: head/predictions/probabilities:0
Method name is: tensorflow/serving/predict
```
Based on our knowledge of `DNNLinearCombinedClassifier`, and the input fields we defined, this looks as we expect. (Notice that the model generates multiple outputs).

### Check local prediction with gcloud

Another useful sanity check is running local prediction with your trained model. We'll use the [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/downloads) command-line tool for that.

We'll use the example input in [`test.json`](test.json) to predict a person's income bracket' based on the features encoded in the `test.json` instance. Again, we point to the directory containing the saved model. 


```sh
gcloud ml-engine local predict --model-dir $output_dir/export/census/<timestamp> --json-instances test.json
```

```
CLASS_IDS  CLASSES  LOGISTIC               LOGITS                 PROBABILITIES
[0]        [u'0']   [0.06585630029439926]  [-2.6521551609039307]  [0.9341437220573425, 0.06585630774497986]
```

You can see how the input fields in `test.json` correspond to the inputs listed by the `saved_model_cli` command above, and how the prediction outputs correspond to the outputs listed by `saved_model_cli`.   
In this model, Class 0 indicates income <= 50k and Class 1 indicates income >50k.

## Using Cloud ML Engine for easy distributed training and scalable online prediction

In the previous section, we looked at how to use `tf.estimator.train_and_evaluate` to train and export a model, and then make predictions using the trained model.

In this section, you'll see how easy it is to use the same code — without any changes — to do **distributed training on Cloud ML Engine (CMLE)**, thanks to the **`Estimator`** class and **`train_and_evaluate`**.  Then we'll use **CMLE Online Prediction** to scalably serve the trained model.

To launch a training job on CMLE, we can again use `gcloud`.  We'll need to package our code so that it can be deployed, and specify the Python file to run to start the training (`--module-name`).  

The `trainer` module is [here](trainer).
`trainer.task` is the entry point, and when that file is run, it calls `tf.estimator.train_and_evaluate`.  
(You can read more about how to package your code [here](https://cloud.google.com/ml-engine/docs/packaging-trainer)).  

If we want to, we could test (distributed) training via `gcloud` locally first, to make sure that we have everything packaged up correctly. See the accompanying [notebook](using_tf.estimator.train_and_evaluate.ipynb) for details.  

But here, we'll jump right in to using Cloud ML Engine (CMLE) to do cloud-based distributed training.

We'll set the training job to use the `SCALE_TIER_STANDARD_1` scale spec.  This [gives you](https://cloud.google.com/ml-engine/docs/training-overview#job_configuration_parameters) one 'master' instance, plus four workers and three [parameter servers](xxx). 


```sh
gcloud ml-engine jobs submit training $JOB_NAME --scale-tier `SCALE_TIER_STANDARD_1` \
    --runtime-version 1.4 --job-dir $GCS_JOB_DIR \
    --module-name trainer.task --package-path trainer/ \
    --region us-central1 \
    -- --train-steps 5000 --train-files $GCS_TRAIN_FILE --eval-files $GCS_EVAL_FILE --eval-steps 100      
```

The cool thing about this is that **we don't need to change our code at all to use this distributed config**.  Our use of the Estimator class in conjunction with the CMLE scale specification allows the distributed training config to be transparent to us — it just works.
Further, we could swap in any of the other predefined scale tiers (say `BASIC_GPU`), or define our own custom cluster, again without any code changes.
For example, we could alternately configure our job to [use a GPU cluster](https://cloud.google.com/ml-engine/docs/using-gpus).

Once your training job is running, you can stream its logs to your terminal, and/or monitor it in the [Cloud Console](https://console.cloud.google.com/mlengine/jobs).


<a href="https://amy-jo.storage.googleapis.com/images/census_train_eval/ml_jobs.png" target="_blank"><img src="https://amy-jo.storage.googleapis.com/images/census_train_eval/ml_jobs.png" width=500/></a>


In the logs, you'll see output from the multiple worker replicas and parameter servers that we utilized by specifying a `SCALE_TIER_STANDARD_1 ` cluster.  In the logs viewers, you can filter on the output of a particular node (e.g. a given worker) if you like.

Once your job is finished, you'll find the exported model under the specified GCS directory, in addition to other data such as model checkpoints.
That exported model has exactly the same signature as the locally-generated model we looked at above, and can be used in just the same ways.

### Scalably serve your trained model with CMLE Online Prediction

You can deploy an exported model to Cloud ML Engine and scalably serve it for **prediction**, using the [CMLE Prediction service](https://cloud.google.com/ml-engine/docs/prediction-overview) to generate a prediction on new data with an easy to use REST API. Here we'll look at CMLE Online Prediction, which [just moved to general availability (GA) status](https://cloud.google.com/blog/big-data/2017/12/bringing-cloud-ml-engine-to-more-developers-with-online-prediction-features-and-reduced-prices); but [batch prediction](https://cloud.google.com/ml-engine/docs/batch-predict) is supported as well.

The online prediction service scales the number of nodes it uses to maximize the number of requests it can handle without introducing too much latency. To do that, the service:

- Allocates some nodes the first time you request predictions after a long pause in requests.
- Scales the number of nodes in response to request traffic, adding nodes when traffic increases, and removing them when there are fewer requests.
- Keeps at least one node ready to handle requests even when there are none to handle. It then scales down to zero by default when your model version goes several minutes without a prediction request (but if you like, you can specify a minimum number of nodes to keep ready for a given model).

See the accompanying [notebook](using_tf.estimator.train_and_evaluate.ipynb) for details on how to deploy your model so that you can use it to make predictions.

Once your model is serving with CMLE Online Prediction, you can access it via a REST API.  It's [easy](https://cloud.google.com/ml-engine/docs/online-predict#requesting_predictions) to do this programmatically via the Google Cloud Client libraries or, via `gcloud`.    
`gcloud` is great for testing your deployed model, and the command looks almost the same as it did for the local version of the model:

```sh
gcloud ml-engine predict --model census --version v1 --json-instances test.json
```

The Cloud Console makes it easy to inspect the different versions of a model, as well as set the default version: [console.cloud.google.com/mlengine/models](https://console.cloud.google.com/mlengine/models).
You can list your model information using `gcloud` too.

<a href="https://amy-jo.storage.googleapis.com/images/census_train_eval/ml_model_details.png" target="_blank"><img src="https://amy-jo.storage.googleapis.com/images/census_train_eval/ml_model_details.png" width=500/></a>


## Summary -- and what's next?

In this example, we've walked through how to configure and use `tf.estimator.train_and_evaluate`.  It enables distributed execution for training and evaluation, while also supporting local execution, and provides consistent behavior for across both local/non-distributed and distributed configurations.

For more, see the accompanying [notebook](using_tf.estimator.train_and_evaluate.ipynb) for information about how to run your training job on a CMLE GPU cluster, and how to use CMLE to do [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/hyperparameter-tuning-overview).




