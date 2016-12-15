import tensorflow as tf
from tensorflow.python.ops import seq2seq
import numpy as np

class Seq2seqModel(object):
    def __init__(
            self, sentence_length, embedding_size, hidden_size, vocabulary_size, cell):

        self.input_x = tf.placeholder(tf.float32, [None, sentence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, sentence_length, embedding_size], name="input_y")
        self.output_y = tf.placeholder(tf.float32, [None, sentence_length, 8000], name="output_y")
        batch_size = tf.shape(self.input_x)[0]
        weights = {
            'out': tf.Variable(tf.random_normal([hidden_size, vocabulary_size]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([vocabulary_size]))
        }
        x = tf.transpose(self.input_x, [1,0,2])
        x = tf.reshape(x, [-1, embedding_size])
        x = tf.split(0, sentence_length, x)
        y = tf.transpose(self.input_y, [1,0,2])
        y = tf.reshape(y, [-1, embedding_size])
        y = tf.split(0, sentence_length, y)

        outputs, states = seq2seq.basic_rnn_seq2seq(x, y, cell)
        self.states = tf.identity(states, name="hidden_state")

        outputs = tf.transpose(outputs, [1,0,2])
        outputs = tf.reshape(outputs, [-1, hidden_size])
        pred = tf.matmul(outputs, weights['out']) + biases['out']
        pred = tf.reshape(pred, [batch_size, sentence_length, vocabulary_size], name="prediction")

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(pred, [-1, 8000]), tf.reshape(self.output_y, [-1, 8000])))
        self.correct_pred = tf.equal(tf.argmax(tf.reshape(pred, [-1, 8000]),1), tf.argmax(tf.reshape(self.output_y, [-1, 8000]),1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
