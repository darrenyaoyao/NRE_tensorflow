import tensorflow as tf
import numpy as np

class SenAutoencoder(object):
    """
    A autoencoder apply on all data included NA and non-NA data.
    """
    def __init__(
      self, sequence_length, embedding_size,
      filter_sizes=3, num_filters=64, l2_reg_lambda=0.0):

        with tf.device('/gpu:1'):
            #Placeholders for input and dropout. (input = output)
            self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            #Keeping track of regularization loss (optional)
            l2_loss = tf.constant(0.0)

            self.embedding_chars_expanded = tf.expand_dims(self.input_x, -1)

            # Build an encoder and create a convolution layer for each filter size
            current_input = self.embedding_chars_expanded
            shape = current_input.get_shape().as_list()
            W = tf.Variable(
                tf.truncated_normal([
                    filter_sizes,
                    embedding_size,
                    1, num_filters], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            encoder = W
            output = tf.nn.relu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1,1,1,1], padding="VALID", name="conv"), b))
            current_input = output

            z = current_input

            #Build an decoder using the same weights
            W = encoder
            b = tf.Variable(tf.constant(0.1, shape=[W.get_shape().as_list()[2]]))
            output = tf.nn.relu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    [64, 100, 52, 1],
                    strides=[1,1,1,1], padding="VALID"), b))
            current_input = output

            y = current_input
            self.loss = tf.reduce_sum(tf.square(y - tf.expand_dims(self.input_y, -1)))
