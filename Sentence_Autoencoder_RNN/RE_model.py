import tensorflow as tf
import numpy as np

class RE_model(object):
    """
    A merge model for relation extraction.
    Combine unsupervised learning and supervised learning
    """
    def __init__(
            self, num_relations, input_size, hidden_size, l2_reg_lambda=0.0):

        #Placeholder for input_x, input_y, dropout_keep_prob
        self.input_x = tf.placeholder(tf.float32, [None, input_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_relations], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        W = {
            'h1': tf.Variable(tf.random_normal([input_size, hidden_size])),
            'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size])),
            'out': tf.Variable(tf.random_normal([hidden_size, num_relations]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_size])),
            'b2': tf.Variable(tf.random_normal([hidden_size])),
            'out': tf.Variable(tf.random_normal([num_relations]))
        }

        layer_1 = tf.add(tf.matmul(self.input_x, W['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, W['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        h_drop = tf.nn.dropout(layer_2, self.dropout_keep_prob)
        self.output = tf.add(tf.matmul(h_drop, W['out']), biases['out'])
        self.prediction = tf.argmax(self.output, 1, name="predictions")

        losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.input_y)
        self.loss = tf.reduce_mean(losses, name="loss")

        correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

