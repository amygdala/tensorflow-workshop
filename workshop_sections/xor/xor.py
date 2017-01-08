import argparse
import math
import os

import numpy as np

import tensorflow as tf


def make_graph(features, labels, num_hidden=8):
  hidden_weights = tf.Variable(tf.truncated_normal(
      [2, num_hidden],
      stddev=1/math.sqrt(2)
  ))

  # Shape [4, num_hidden]
  hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights))

  output_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, 1],
      stddev=1/math.sqrt(num_hidden)
  ))

  # Shape [4, 1]
  logits = tf.matmul(hidden_activations, output_weights)

  # Shape [4]
  predictions = tf.sigmoid(tf.squeeze(logits))
  loss = tf.reduce_mean(tf.square(predictions - tf.to_float(labels)))

  gs = tf.Variable(0, trainable=False)
  train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss, global_step=gs)

  return train_op, loss, gs

def main(output_dir, checkpoint_every, num_steps):
  graph = tf.Graph()

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with graph.as_default():
    features = tf.placeholder(tf.float32, shape=[4, 2])
    labels = tf.placeholder(tf.int32, shape=[4])

    train_op, loss, gs, update_acc = make_graph(features, labels)
    init = tf.global_variables_initializer()

  with tf.Session(graph=graph) as sess:
    init.run()
    step = 0
    xy = np.array([
        [True, False],
        [True, True],
        [False, False],
        [False, True]
    ], dtype=np.float)
    y_ = np.array([True, False, False, True], dtype=np.int32)
    while step < num_steps:

      _, step, loss_value = sess.run(
          [train_op, gs, loss],
          feed_dict={features: xy, labels: y_}
      )
    print('Final loss is: {}'.format(loss_value))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-steps', type=int, default=5000)
  parser.add_argument(
      '--output-dir',
      type=os.path.abspath,
      default=os.path.abspath('output')
  )
  args = parser.parse_args()
  main(args.output_dir, args.checkpoint_every, args.num_steps)
