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

# This file is a modification of the code here:
# https://github.com/dennybritz/cnn-text-classification-tf

import numpy as np
import re
import itertools
from collections import Counter

import json
import tensorflow as tf

vocabulary_mapping = None
vocabulary_inv = None
#sequence_length_all = None


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(cat1="./data/reddit/aww/subreddit_news",
        cat2="./data/reddit/aww/subreddit_aww"):
    """
    Loads two-category data from files, splits the data into words and generates
    labels. Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = list(open(cat1, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = list(open(cat2, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


# aju
def build_vocab_mapping(run="", write_mapping=True,
        cat1="./data/reddit/aww/subreddit_news",
        cat2="./data/reddit/aww/subreddit_aww"):
    """
    Generate vocabulary mapping info, save it for later eval. This ensures that
    the mapping used for the eval is the same.
    """
    global vocabulary_mapping
    global vocabulary_inv
    # Load data from files
    positive_examples = list(open(cat1, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(cat2, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    vocabulary_mapping, vocabulary_inv = build_vocab(
        pad_sentences(x_text))
    vocab_file = "vocab{}.json".format(run)
    if write_mapping:
        with open(vocab_file, "w") as f:
            f.write(json.dumps(vocabulary_mapping))


def pad_sentences(sentences, padding_word="<PAD/>",
        max_sent_length=40):
    """
    Pads all sentences to the same length. The length is defined by the min of
    the longest sentence and a given max sentence length.
    Returns padded sentences.
    """
    #global sequence_length_all
    global vocabulary_mapping

    sequence_length = max(len(x) for x in sentences)
    # aju - try capping sentence length to reduce amount of padding necessary
    # overall.
    print("setting seq length to min of %s and %s" % (sequence_length, max_sent_length))
    sequence_length = min(sequence_length, max_sent_length)
    print("capped longest seq length: %s" % sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        # aju - truncate as necessary
        sentence = sentences[i][:sequence_length]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, max_vocab=10000):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # aju - cap vocabulary
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word. Use the 'max_vocab' most common.
    vocabulary_inv = [x[0] for x in word_counts.most_common(max_vocab)]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

# aju
def get_embeddings(vocab_size, embedding_size, emb_file):  # expected sizes
    """..."""
    global vocabulary_mapping
    # create a matrix of the right size
    embeddings = np.random.uniform(
        -1.0,1.0,size=(vocab_size, embedding_size)).astype('float32')
    # get the vocabulary mapping info
    if not vocabulary_mapping:
        # should have already generated the vocab mapping
        print("Don't have vocabulary mapping.")
        return None
    vocabulary = vocabulary_mapping
    if len(vocabulary) != vocab_size:
        print('vocab size mismatch: %s vs %s' % (vocab_size, len(vocabulary)))
        return None
    # read and parse the generated embeddings file
    try:
        with open(emb_file, "r") as f:
            for line in f:
                edict = json.loads(line)
                key = list(edict.keys())[0]
                # see if key is in the vocab
                if key in vocabulary:
                    # then add the embedding vector
                    emb = edict[key][0]
                    if len(emb) != embedding_size:
                        print(
                            "embedding size mismatch for word {}: {} vs {}".format(
                                key, embedding_size, len(emb)))
                        return None
                    vocab_idx = vocabulary[key]
                    embeddings[vocab_idx] = emb
        return tf.convert_to_tensor(embeddings)
    except Exception as e:
        print(e)
        return None


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """

    # aju - with reduced vocab, need to account for word not present in
    # vocab -- what if I just use the padding word?
    # TODO -- pass padding word in as an arg
    padding_word="<PAD/>"
    pad_idx = vocabulary[padding_word]
    x = np.array([[vocabulary.get(word, pad_idx) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(run="", eval=False, vocab_file=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    global vocabulary_mapping
    global vocabulary_inv

    print("eval mode: {}".format(eval))
    print("vocab file: {}".format(vocab_file))
    # Load and preprocess data
    # aju
    if eval:
        #build_vocab_mapping(write_mapping=False)
        print("loading generated vocab mapping")
        with open(vocab_file, "r") as f:
            mapping_line = f.readline()
            vocabulary_mapping = json.loads(mapping_line)
        #sentences, labels = load_data_and_labels()
    else:
        build_vocab_mapping(run=run)
    print("loading training data")
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary_mapping)
    return [x, y, vocabulary_mapping, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
