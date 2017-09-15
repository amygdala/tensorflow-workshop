
# Run and modify a version of MNIST built using TensorFlow's high-level APIs

  - [Running the example as a script](#running-the-example-as-a-script)
  - [Running the example in a Jupyter notebook](#running-the-example-in-a-jupyter-notebook)
  - [TensorBoard](#tensorboard)
  - [Change the learning rate of your classifier](#change-the-learning-rate-of-your-classifier)
  - [Compare the DNNClassifier's performance on MNIST to a LinearClassifier](#compare-the-dnnclassifiers-performance-on-mnist-to-a-linearclassifier)

This lab uses TensorFlow's high-level APIs, in `tf.estimator`, in order to easily build a NN with hidden layers. It uses [`DNNClassifier`](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier).
As part of the lab, we'll explore what [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) can do.

The lab also includes a demonstration of using a [LinearClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier).

## Running the example as a script

The python script for this section is: [`mnist_estimator.py`](./mnist_estimator.py).
Start the lab by running it like this:

```sh
$ python mnist_estimator.py
```

By default, it writes summary and model checkpoint info to a (timestamp-based) subdir
of `/tmp/tfmodels/mnist_estimators`.

## Running the example in a Jupyter notebook

You can also run the example as a Jupyter (ipython) notebook.
Start up the Jupyter server from this directory like this, if it's not already running:

```sh
$ jupyter notebook
```

When the server starts up, it may bring up a tab automatically in your browser. If not, visit
`http://localhost:8888/`.  Then select `mnist_estimator.ipynb`.

## Using 'fashion MNIST' instead of 'regular' MNIST for your dataset

Fashion-MNIST is a direct drop-in replacement for the original MNIST dataset.

<img src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png" width=600 />

You can read more about it, and why it was created, [here](https://github.com/zalandoresearch/fashion-mnist). It is a more challenging dataset than 'regular' MNIST, which has become overused, and is too easy.

Download Fashion-MNIST [here](https://github.com/zalandoresearch/fashion-mnist#get-the-data), and put the files in a known directory. Then, find the line in your code which reads in the MNIST datasets, which will look something like this:

```python
DATA_SETS = input_data.read_data_sets(DATA_DIR)
```

If you're running the notebook, point `DATA_DIR` to the Fashion-MNIST directory you just created. If you're running the python script version, set the `--data_dir` command-line arg to this directory.

## TensorBoard

After the script/notebook runs, or while it's running, start up TensorBoard as follows in a new terminal window. (If you get a 'not found' error, make sure you've activated your virtual environment in that new window):

```sh
$ tensorboard --logdir=/tmp/tfmodels/mnist_estimators
```

Note: This tells TensorBoard to grab summary information from all directories under `/tmp/tfmodels/mnist_estimators`.  So, we can do multiple runs, and compare the results. TensorBoard will automatically pull in data on additional runs as it is added.

Once TensorBoard is running, then in your browser, visit the address indicated (probably `localhost:6006`).

## Change the learning rate of your classifier

Next, edit `mnist_estimator.py` to change the learning rate used by the optimizer, as defined in the
`DNNClassifier` initialization method. **Change the learning rate from 0.1 to 0.5**.
Then rerun the script.

Once the script is running, revisit the running TensorBoard server in your browser. You might want to reload to speed up the display of the new data.  You should see info on at least two different runs being displayed.
You can remind yourself of which is which by checking the model directory path (output to STDOUT) for each training run.

See whether the different learning rates caused different 'eval' results.

## Compare the DNNClassifier's performance on MNIST to a LinearClassifier

In `mnist_estimator.py` you may have noticed that a linear classifier is also defined, using the
[`LinearClassifier`](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier) class.  This class defines a model that is essentially the same as that defined in
`mnist_simple.py`, which we explored in our previous lab-- it does not use any hidden layers.

Try uncommenting `run_linear_classifier()` in `main()` to run it. (You may want to also comment out
`run_dnn_classifier()`).  You can see from STDOUT that the linear classifier is writing its summary information to its own /tmp directory.

Then, try comparing the linear classifier's performance to that of the DNN classifier.
To do this, you can point tensorboard to a *set* of directories, like this (replacing `<path1>` and `<path2>` with the appropriate paths):

```sh
tensorboard --logdir=name1:<path1>,name2:<path2>
```

If you've done runs using both 'regular' MNIST and training MNIST, it is interesting to compare those as well.



