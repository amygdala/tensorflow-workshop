
# Run and modify a version of MNIST built using TensorFlow's high-level APIs

This lab uses TensorFlow's high-level APIs, in `tf.contrib.learn`, in order to easily build a NN with hidden layers. It uses [`DNNClassifier`](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#DNNClassifier).
As part of the lab, we'll explore what [TensorBoard](https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html) can do.

It also includes a bonus demonstration of using a [LinearClassifier](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#LinearClassifier).

The python script for this section is: [`mnist_tflearn.py`](./mnist_tflearn.py).
Start the lab by running it like this:

```shell
$ python mnist_tflearn.py
```

By default, it writes summary and model checkpoint info to a (timestamp-based) subdir of `/tmp/tfmodels/mnist_tflearn`.

After the script runs, or while it's running, start up TensorBoard as follows in a new terminal window. (If you get a 'not found' error, make sure you've activated your virtual environment in that new window):

```shell
$ tensorboard --logdir=/tmp/tfmodels/mnist_tflearn
```

Note: This tells TensorBoard to grab summary information from all directories under `/tmp/tfmodels/mnist_tflearn`.  So, we can do multiple runs, and compare the results -- TensorBoard will automatically pull in data on additional runs as it is added.

Once TensorBoard is running, then in your browser, visit the address indicated (probably `localhost:6060`).

[At this point in the lab, we'll take some time to explore TensorBoard and see what kind of information it can display].

Next, edit `mnist_tflearn.py` to change the learning rate used by the optimizer, as defined in the
`DNNClassifier` initialization method. Change the learning rate from 0.1 to 0.5.
Then rerun the script.

Once the script is running, revisit the running TensorBoard server in your browser. You might want to reload to speed up the display of the new data.  You should see info on at least two different runs being displayed.
You can remind yourself of which is which by checking the model directory path (output to STDOUT) for each training run.

See whether the different learning rates caused different 'eval' results.

## Bonus: Comparing the DNNClassifier's performance on MNIST to a LinearClassifier

In `mnist_tflearn.py` you may have noticed that a linear classifier is also defined, using the
[`LinearClassifier`](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#LinearClassifier) class.  This class defines a model that is essentially the same as that defined in
`mnist_simple.py`, which we explored in our previous lab-- it does not use any hidden layers.

Try uncommenting `run_linear_classifier()` in `main()` to run it. (You may want to also comment out
`run_dnn_classifier()`).  You can see from STDOUT that the linear classifier is writing its summary information to its own /tmp directory.

Then, try comparing the linear classifier's performance to that of the DNN classifier.
To do this, you can point tensorboard to a set of directories, like this (replacing `<path1>` and `<path2>` with the appropriate paths):

```shell
tensorboard --logdir=name1:<path1>,name2:<path2>
```




