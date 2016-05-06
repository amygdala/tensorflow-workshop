
# Introducing word2vec

In this section, we'll take a look at the ['basic' variant of word2vec](xxx) used for [this Tensorflow tutorial](xxx).

## Start the process of training the model

First, double check that you have installed `matplotlib`, as in the installation instructions.
Then, start training the model. Change to the `<tensorflow>/examples/tutorials/word2vec` directory.  Run:

```sh
$ python word2vec_basic.py
```

While it's running, we'll look at what how this graph is constructed, what it does and why that is interesting.

## (Alternately) run the training within a jupyter notebook

[** add explanation **] 

```sh
$ cd <tensorflow>/tensorflow/examples/udacity
$ jupyter notebook
```

Then select the `5_word2vec` notebook from the list.


## Look at the results

After the training is finished, the script will map the model's learned word vectors into a 2D space, and plot the results using matplotlib in conjunction with an [sklearn](xxx) library called [TSNE](xxx).
It will write the plot to an image file named `tsne.png` in the same directory.

<a href="https://amy-jo.storage.googleapis.com/images/tf-workshop/tsne.png" target="_blank"><img src="https://amy-jo.storage.googleapis.com/images/tf-workshop/tsne.png" width="300"/></a>

[** more on what the image shows **]


[** Note/TBD: This script doesn't save the model info.  We could potentially add that and have them fire up tensorboard here.
However, I am inclined just to wait until they run text-cnn to intro tensorboard, since that saves summaries and makes the dashboard quite interesting.
Regardless - we could create a version of basic_word2vec that saves the model info, and just show them a tensorboard rendering in the slides. **]



