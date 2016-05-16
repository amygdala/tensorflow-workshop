
# Introducing word2vec

In this section, we'll take a look at the ['basic' variant of word2vec](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) used for [this Tensorflow tutorial](https://www.tensorflow.org/versions/r0.8/tutorials/word2vec/index.html#vector-representations-of-words).

We'll launch the training process, then while that is running, we'll look at how this graph is constructed, and what it does.

## Start the process of training the model

First, confirm that you can import `matplotlib` and `sklearn`, as in the installation instructions.
Then, start training the model. You can do this in two different ways: from the command line, and from a Jupyter notebook.

### Training from the command line

Run the [`word2vec_basic_summaries.py`](word2vec_basic_summaries.py) script from this directory. (This file is the same as [`word2vec_basic`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py), but has a few additional lines of code that write some summary information to [TensorBoard](https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html)).

```sh
$ python word2vec_basic_summaries.py
```


## (Alternately) train from a Jupyter notebook.

The TensorFlow repo includes a nice Jupyter notebook version of this same model (without the addition of the code that writes the summary info).


```sh
$ cd <tensorflow>/tensorflow/examples/udacity
$ jupyter notebook
```

Then select the `5_word2vec` notebook from the list.
Step through its setup and graph definition steps until you reach the training step, and start it running.

## The word2vec_basic model graph

While the model is training, we'll walk through the code, look at how its graph is constructed, what it does, and why that is interesting.

<a href="https://storage.googleapis.com/oscon-tf-workshop-materials/images/word2vec_basic.png" target="_blank"><img src="https://storage.googleapis.com/oscon-tf-workshop-materials/images/word2vec_basic.png" width="500"/></a>

This example picks a random set of words from the top 100 most frequent, and periodically outputs the 'nearby' results for those words.  You can watch the set for each word becoming more accurate (mostly :) as the training continues. To get really impressive results, you'll need to run for more than the default number of steps.

## Look at the results

After the training is finished, the script will map the model's learned word vectors into a 2D space, and plot the results using `matplotlib` in conjunction with an `sklearn` library called
[TSNE](https://lvdmaaten.github.io/tsne/).
It will write the plot to an image file named `tsne.png` in the same directory.

<a href="https://amy-jo.storage.googleapis.com/images/tf-workshop/tsne.png" target="_blank"><img src="https://amy-jo.storage.googleapis.com/images/tf-workshop/tsne.png" width="500"/></a>

In your projection plot, you should see similar words clustered close to each other.

### Take a look at TensorBoard

If you ran the command line version, which writes some information to TensorBoard, you can start it up like this to take a look:

```sh
$ tensorboard tensorboard --logdir=/tmp/word2vec_basic/summaries
```

We will dive into more detail on TensorBoard in a later section of the workshop.

## Excercise: find the 'nearby' words for a specific given word

See if you can figure out how to modify the [`word2vec_basic_summaries.py`](word2vec_basic_summaries.py) code to evaluate and output the 'nearby' set for a specific word too.

E.g., picking "government" as the word will give a result like this (after about 500K training steps):

```
Nearest to b'government': b'governments', b'leadership', b'regime', b'crown', b'rule', b'leaders', b'parliament', b'elections',
```

After you've given it a try yourself, [`word2vec_basic_nearby.py`](word2vec_basic_nearby.py) shows one simple way to do this.


