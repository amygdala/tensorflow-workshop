import tensorflow as tf


def get_inputs(filenames, batch_size, num_epochs, sentence_length):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read_up_to(
            filename_queue, batch_size)

        features = tf.parse_example(
            serialized_examples,
            {
                'words': tf.FixedLenFeature([sentence_length], tf.int64),
                'score': tf.FixedLenFeature([1], tf.int64)
            }
        )

        scores = tf.reshape(features['score'], [-1])
        sentences, scores = tf.train.shuffle_batch(
            [features['words'], scores],
            batch_size,
            capacity=1000 + 3 * batch_size,
            min_after_dequeue=1000,
            enqueue_many=True
        )
    return sentences, scores


def make_model_fn(args):
  def _model_fn(words,  # Shape [batch_size, sentence_length]
                subreddits,  # Shape [batch_size]
                mode):
    def partitioner(shape, **unused_args):
        partitions_list = [1] * len(shape)
        partitions_list[0] = min(args.num_param_servers, shape[0].value)
        return partitions_list

    batch_size, sentence_length = words.get_shape().as_list()

    scores_cast = tf.cast(scores, tf.float32)
    # Shape [vocab_size, embedding_size]
    with tf.variable_scope('embeddings',
                           partitioner=partitioner):
        embeddings = tf.get_variable(
            'embeddings',
            shape=[args.vocab_size, args.embedding_size]
        )

    # Shape [batch_size, sentence_length, embedding_size]
    word_embeddings = tf.nn.embedding_lookup(embeddings, words)

    lstm = tf.nn.rnn_cell.LSTMCell(
        args.lstm_size, num_proj=1, state_is_tuple=True)

    outputs, state = tf.nn.rnn(
        lstm,
        [word_embeddings[:, i, :] for i in range(sentence_length)],
        dtype=tf.float32,
        scope='rnn'
    )

    # Shape [sentence_length, batch_size]
    outputs_concat = tf.squeeze(tf.pack(outputs))

    # Shape [batch_size]
    predictions = tf.reduce_mean(
        tf.exp(outputs_concat), reduction_indices=[0])

    tf.histogram_summary('predictions', predictions)

    self.loss = tf.reduce_mean(tf.square(predictions - scores_cast))

    train_op = tf.contrib.layers.optimize_loss(
        self.loss,
        self.global_step,
        learning_rate,
        tf.train.AdamOptimizer,
        clip_gradients=1.0,
    )
