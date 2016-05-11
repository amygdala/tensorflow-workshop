
[** TBD.

Include showing tensorboard, as this example is well set up for it-- it generates 'summary events' for both train and dev-test.

Scripts are modifications of code from: https://github.com/dennybritz/cnn-text-classification-tf 

The graph for this model:

<a href="https://storage.googleapis.com/oscon-tf-workshop-materials/images/text-cnn-graph.png" target="_blank"><img src="https://storage.googleapis.com/oscon-tf-workshop-materials/images/text-cnn-graph.png" width="300"/></a>


Processed reddit data: 'news' & 'aww' subreddits:
https://pantheon.corp.google.com/storage/browser/oscon-tf-workshop-materials/processed_reddit_data/news_aww/?project=oscon-tf-workshop

Embeds (generated from the 'text8' data): https://pantheon.corp.google.com/m/cloudstorage/b/aju-vtests2-oscon/o/all_embeddings.json

**]

```sh
$ tensorboard --logdir=runs
```

[** add tensorboard screenshots **] 