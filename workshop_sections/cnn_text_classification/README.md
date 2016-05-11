
[** TBD. **]

In slides, talk first about convolutional NNs.
Will include showing tensorboard and summarywriter, as this example is well set up for it.

Scripts are modifications of code from: https://github.com/dennybritz/cnn-text-classification-tf

## Using convolutional NNs for text classification, and TensorBoard

[** to be fleshed out **]

- Start a training run that does not use the embeddings
- In a separate terminal window, start up tensorboard

```sh
$ tensorboard --logdir=runs
```

- Go back to the code.  Walk through how the graph is constructed at a high level.
- Look at the graph in tensorboard. Show a bit re: how to navigate around it.
- Go back to the code, walk through how the SummaryWriter is being used.

## Using convolutional NNs for text classification, part II: using learned word embeddings

[** to be fleshed out **]

- stop the training run above, but keep tensorboard running
- download the generated embeddings file (hopefully the reddit version)
- start up another training run that uses the learned embeddings to initialized the embedding vectors.  Talk about why this is interesting. [Add screenshots from the results of my runs].
- Look at the results in tensorboard after this second training run has been running for a while. The initial benefits of initializing with the learned embeddings should be quickly obvious.

[** Notes:

The graph for this model (click for larger version):

<a href="https://storage.googleapis.com/oscon-tf-workshop-materials/images/text-cnn-graph.png" target="_blank"><img src="https://storage.googleapis.com/oscon-tf-workshop-materials/images/text-cnn-graph.png" width="300"/></a>


Processed reddit data: 'news' & 'aww' subreddits [specifics to be added; some of these files are aggregations of others]:
https://pantheon.corp.google.com/storage/browser/oscon-tf-workshop-materials/processed_reddit_data/news_aww/?project=oscon-tf-workshop

Embeds generated from the 'text8' data: https://pantheon.corp.google.com/m/cloudstorage/b/aju-vtests2-oscon/o/all_embeddings.json

Embeds generated from the reddit 'aww' and 'news' subreddit data: xxx

**]

```sh
$ tensorboard --logdir=runs
```


[** Look at SummaryWriter... consider exercise where they change the summary info. **]

[** Add tensorboard screenshots **]
