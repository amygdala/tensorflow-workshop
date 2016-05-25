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

import tensorflow as tf


class DistributedTextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling
    and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, cluster_def,
                 l2_reg_lambda=0.0, wtensor=None, worker_job='worker',
                 param_server_job='ps', master_device='/job:worker/task:0'):

        # Placeholders for input, output and dropout

        workers = ['/job:{}/task:{}'.format(worker_job, i)
                   for i in range(len(cluster_def[worker_job]))]
        param_servers = ['/job:{}/task:{}'.format(param_server_job, i)
                         for i in range(len(cluster_def[param_server_job]))]

        if wtensor is None:
            wtensor = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)

        with tf.device(master_device):
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            input_x_minibatches = tf.split(0, len(workers), self.input_x)
            self.input_y =  tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            input_y_minibatches = tf.split(0, len(workers))
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device(param_servers[0]):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

        embeddings = []
        for param_server in param_servers:
            with tf.device(param_server):
                embeddings.append(tf.Variable(wtensor, name="W"))

        filter_weights = []
        filter_biases = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device(param_servers[i % len(param_servers)]):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # Convolution Layer
                filter_weights.append(tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name="W"))
                filter_biases.append(tf.Variable(
                    tf.constant(0.1, shape=[num_filters]), name="b"))

        losses = []
        accuracies = []
        for j, worker in enumerate(workers):
            with tf.device(worker):
                embedded_chars = tf.nn.embedding_lookup(
                    embeddings, input_x_minibatches[j])
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    # Create a convolution + maxpool layer for each filter size
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        conv = tf.nn.conv2d(
                            embedded_chars_expanded,
                            filter_weights[i],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv"
                        )
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, filter_biases[i]), name="relu")
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool"
                        )
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                h_pool = tf.concat(3, pooled_outputs)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(
                        h_pool_flat, self.dropout_keep_prob)

                # Final (unnormalized) scores and predictions
                with tf.name_scope("output"):
                    W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                    predictions = tf.argmax(scores, 1, name="predictions")

                # CalculateMean cross-entropy loss
                with tf.name_scope("loss"):
                    loss = tf.nn.softmax_cross_entropy_with_logits(scores, input_y_minibatches[j])
                    losses.append(tf.reduce_mean(loss) + l2_reg_lambda * l2_loss)

                # Accuracy
                with tf.name_scope("accuracy"):
                    correct_predictions = tf.equal(predictions, tf.argmax(input_y_minibatches[j], 1))
                    accuracies.append(tf.reduce_mean(
                        tf.cast(correct_predictions, "float"), name="accuracy"))

        with tf.device(master_device):
            self.loss = tf.add_n(losses) / tf.convert_to_tensor(len(workers))
            self.accuracy = tf.add_n(accuracies) / tf.convert_to_tensor(len(workers))

