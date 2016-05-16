# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import time
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify {}. \
                Can you get to it with a browser?'.format(
                    filename
                )
            )
    return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(
        batch[i],
        reverse_dictionary[batch[i]],
        '->',
        labels[i, 0],
        reverse_dictionary[labels[i, 0]]
    )

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


import ast

cluster_config = ast.literal_eval(os.environ.get('CLUSTER_CONFIG'))
cluster_spec = tf.train.ClusterSpec(cluster_config)
num_workers = len(cluster_config['worker'])
num_param_servers = len(cluster_config['ps'])
master_device = '/job:jupyter/task:0'

graph = tf.Graph()
with graph.as_default():
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device(master_device):
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Make one minibatch of data for each worker
        train_inputs_list = tf.split(0, num_workers, train_inputs)
        train_labels_list = tf.split(0, num_workers, train_labels)

    # Create a variable embedding for each parameter server
    for i in range(num_param_servers):
        embeddings = []
        with tf.device('/job:ps/task:{}'.format(i)):
            # Look up embeddings for inputs.
            embeddings.append(
                tf.Variable(tf.random_uniform(
                    [vocabulary_size, embedding_size],
                    -1.0,
                    1.0
                ))
            )

    with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        global_step = tf.Variable(0)

    train_ops = []
    losses = []
    summaries = []
    # Assign computational tasks to each worker
    for i in range(num_workers):
        with tf.device('/job:worker/task:{}'.format(i)):
            embed = tf.nn.embedding_lookup(embeddings, train_inputs_list[i])
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative
            # labels each time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                               num_sampled, vocabulary_size))
            losses.append(loss)
            # Construct the SGD optimizer using a learning rate of 0.1.
            train_ops.append(
                tf.train.GradientDescentOptimizer(0.1).minimize(
                    loss, global_step=global_step))

    with tf.device(master_device):
        optimizer = tf.group(*train_ops)
        summary_op = tf.merge_summary(summaries)
        init_op = tf.initialize_all_variables()
        losses_sum = tf.add_n(losses)
        # Compute the cosine similarity between minibatch examples
        # and the embeddings
        norm = tf.sqrt(tf.reduce_sum(
            tf.square(embeddings[0]), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
        summaries.append(
            tf.scalar_summary('loss-{}'.format(i), loss))


# Step 5: Begin training.
num_steps = 100001

with tf.Session('grpc://localhost:8080', graph=graph) as session:
    init_op.run()
    print("Initialized")

    average_loss = 0
    step = 0
    timestamp = str(int(time.time()))
    log_dir = '/var/log/tensorflow/word2vec_basic/summaries/{}'.format(
        timestamp)
    summary_writer = tf.train.SummaryWriter(log_dir, graph=graph)

    while step < num_steps:
        step = global_step.eval()

        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op
        # Also evaluate the training summary op
        # Average the workers' losses, and advance the global step
        _, summary, average, step = session.run(
            [optimizer, summary_op, average, global_step], feed_dict=feed_dict)

        summary_writer.add_summary(summary, global_step=step)
        average_loss += average

        if not step == 0 and step % 1500 == 0:
            average_loss /= 1500
            # The average loss is an estimate of the loss over the last 2000
            # batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                    print(log_str)

    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
