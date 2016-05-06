
# A less basic version of word2vec

In this section of the workshop, we're going to look at a more optimized version of the word2vec code.

This directory contains a slightly modified version of the `word2vec_optimized.py` file in the `models/embedding` [directory](xxx) of the TensorFlow repo.
See that directory's README for further detail.

The script in this directory has been modified from the original to allow a mode that doesn't run the model training, but instead just restores the model from saved checkpoint info, then starts up an interactive shell to let you explore the results.
(We'll do that below).

## Start the process of training your model

First, as indicated in `word2vec_optimized.py`, download and unzip this file: `http://mattmahoney.net/dc/text8.zip`.
Then, create a `saved_model` directory in which to store the saved model info. (You can put this directory whereever you like).

Start the model training with the following command. In the flags, specify the location of the unzipped `text8` file, and the directory you just created in which you're going to save your model info.


```sh
$ python word2vec_optimized.py --train_data=text8 --eval_data=questions-words.txt --save_path=/tmp --train=true --epochs_to_train=2
```

Due to the workshop time contraints, we're just training for 2 epochs here. (This won't properly train the model).
Let's take a quick look at the code while it's running.

[** Notes: do we run this on tensorkubes?  If so, how long does it take?
If running locally, we won't have time to fully train it.  We will just need to tell them to ctl-c at some point. Then below, they use an already-generated model.
Also: `word2vec_optimized.py` doesn't checkpoint regularly -- it just saves the model at the end.  Maybe we want to modify this version of the script to do that too, so that when they run training, they can look at the saver output.  **]

## Load and use the saved model results

We can save a graph's structure and values to disk -- both while it is training, to checkpoint, and when it is done training, to later use the final result.

`word2vec_optimized` shows how to restore graph variables from disk: we'll restore the model with checkpointed learned variables. (A later example will show how to load the graph structure from disk as well).

Because you won't have time to fully train your model during this workshop, download pregenerated checkpoint data from an already-trained model [here](xxx).
Then, create a `saved_model` directory in which to store the saved model info. (This example creates a subdirectory of the current directory; however, you can put it where you like.)

Unzip the [** xxx file **] into this directory.  You should see some `model.ckpt*`` files as well as a file named `checkpoint`.

Then, run the following command, this time pointing to the directory in which you put the saved model info (instead of `/tmp` like you did above). Don't include the training data or the `--train=true` flag this time.

```sh
$ python word2vec_optimized.py --eval_data=questions-words.txt --save_path=saved_model
```

Without the `--train` flag, the script builds the model graph, and then restores the saved model variables to it.

### Use the trained model to do word relationship analyses

[** in interactive shell, with model loaded, run model.nearby() and model.analog() methods.  Talk through how we're running the graph to generate this info. They will come back to this concept when we have them generate embeddings. **]

### Use the trained model to get the learned word embeddings

[** use `word2vec_optimized_embeds.py`, which has a placeholder for where they should edit. **] 

[** tbd: make this an exercise: define most of the function, make them add the code that actually runs the graph to get a word embedding. The answer is here: `word2vec_optimized_embeds_answ.py`. **]

## Coming up soon: using the learned embeddings

In a following section, we'll use the word vectors (the embeddings) learned by this model, to improved the performance of different model -- a convolutional NN for text classification.
