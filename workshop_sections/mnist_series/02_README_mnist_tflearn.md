
# Run and modify a version of MNIST built using TensorFlow's high-level APIs

This lab uses TensorFlow's high-level APIs, in `tf.contrib.learn`, in order to easily build a NN with hidden layers, using the `DNNClassifier` class.

[** TBD: 
- Use a different optimizer, the "Adam" optimizer, and compare. 
  Then compare the two runs in TensorBoard by starting it like this:
  `tensorboard --logdir=/tmp/tfmodels/mnist_tflearn`

- Optional: Try uncommenting the linear classifier essentially the same as mnist_simple, and compare. 
optimizer solution: add to the DNNClassifier init: 

**]

The python script for this section is: [`mnist_tflearn.py`](./mnist_tflearn.py).
Run it like this:

```shell
$ python mnist_tflearn.py
```

After you've run the script, start up [TensorBoard](https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html) to see the training progression.
To do this, start a new terminal window, and start up your virtual environment as necessary.
Then run the following command to point TensorBoard to the directory containing the training summary information:

```shell
$ tensorboard --logdir=/tmp/tfmodels
```

Once it's running, then in your browser, visit the address indicated (probably `localhost:6060`).


