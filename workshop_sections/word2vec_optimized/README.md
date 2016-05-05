
# A less basic version of word2vec

In this section of the workshop, we're going to look at a more optimized version of the word2vec code.

Change to the `./word2vec` directory.
This directory contains a slightly modified version of the `word2vec_optimized.py` file in the `models/embedding` [directory](xxx) of the TensorFlow repo.

The code has been modified from the original to allow a mode that doesn't run the training, but just looks for 



## Start the process of training your model

Start the training process like this:

```sh
$ xxxx
```

[** do we run this on tensorkubes?  If so, how long does it take?]

[** If running locally, we won't have time to fully train it **]

## Load and use the saved model results

We can save a graph's structure and values to disk -- both while it is training, in order to checkpoint, and when it is done training, to save and later use the final result.

In `word2vec_optimized`, we'll look at how to restore graph variables from disk: we'll first build the same graph structure that we trained, then restore the checkpointed learned variables. (A later example will show how to load the graph structure from disk as well).

Because you won't have time to fully train your model, download checkpoint info from an already-trained model [here](xxx).
[** Probably they will need to do this. **]

Then, run:

```sh
$ xxxx
```

Without the `--train` flag, the script builds the model graph, and then restores it [** .. **]

[** point out interactive shell **]


## Coming up - using the learned embeddings

Then, in a following section, we'll use the word vectors (the embeddings) learned by this model, to improved the performance of different model -- a convolutional NN for text classification.