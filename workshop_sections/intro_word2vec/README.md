
# Introducing word2vec

In this section, we'll take a look at the ['basic' variant of word2vec](xxx) used for [this Tensorflow tutorial](xxx).

## Start the training process

First, start training the model. Change to the `<tensorflow>/examples/tutorials/word2vec` directory.  Run:

```sh
$ python word2vec_basic.py
```

While it's running, we'll look at what how this graph is constructed, what it does and why that is interesting.

## Look at the results



[** Note/TBD: This script doesn't save the model info.  We could potentially add that and have them fire up tensorboard here.
However, I am inclined just to wait until they run text-cnn to intro tensorboard, since that saves summaries and makes the dashboard quite interesting.
Regardless - we could create a version of basic_word2vec that saves the model info, and just show them a tensorboard rendering in the slides. **]



