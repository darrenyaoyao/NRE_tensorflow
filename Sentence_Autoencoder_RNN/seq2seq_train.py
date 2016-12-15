import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.python.ops import rnn, rnn_cell
from DataManager import DataManager
from seq2seq_model import Seq2seqModel
from tensorflow.python.util import nest

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("sentence_length", 80, "Each sentence length (default: 80)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

dataManager = DataManager()

# Load data
print("Loading training data...")
x_text, _, y_num = dataManager.load_training_data()
print("Finish loading data")

#Prepare data
x = []
for data in x_text:
    a = 80-len(data)
    if a > 0:
        padding = [np.zeros(52) for i in range(a)]
        x.append(data+padding)
    else:
        x.append(data[:80])

def decoder_y(x):
    x = np.asarray(x)
    y = []
    for i in x:
        y.append([np.zeros(52)]+list(i)[:-1])
    return y

def one_hot(y):
    y = list(y)
    y_l = []
    for i, s in enumerate(y):
        a = 80-len(s)
        if a > 0:
            padding = [0 for k in range(a)]
            y[i] = y[i]+padding
        else:
            y[i] = y[i][:80]
        for j in range(80):
            v = np.zeros(8000, dtype=np.float32)
            v[y[i][j]] = 1.0
            y[i][j] = v
        y_l.append(np.asarray(y[i], dtype=np.float32))
    return np.asarray(y_l, dtype=np.float32)

y_label = y_num

x = np.asarray(x)
y_label = np.asarray(y_label)

#Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(x)))
x = x[shuffle_indices]
y_label = y_label[shuffle_indices]

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config=session_conf) as sess:
        lstm_cell = rnn_cell.BasicLSTMCell(128, forget_bias=1.0)
        seq2seq_model = Seq2seqModel(
                sentence_length = 80,
                embedding_size = 52,
                hidden_size = 128,
                vocabulary_size = 8000,
                cell = lstm_cell)

        #Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(seq2seq_model.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "seq2seq_runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        #Summaries for cost and accuracy
        loss_summary = tf.scalar_summary("cost", seq2seq_model.cost)
        acc_summary = tf.scalar_summary("accuracy", seq2seq_model.accuracy)

        #Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        #Checkpoint directory. Tensorflow assume this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        #Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

        def train_step(x_batch, y_batch, y_label_batch):
            #Single training step
            feed_dict={
                seq2seq_model.input_x: x_batch,
                seq2seq_model.input_y: y_batch,
                seq2seq_model.output_y: y_label_batch
            }
            _, step, summaries, cost, accuracy, = sess.run(
                [train_op, global_step, train_summary_op, seq2seq_model.cost, seq2seq_model.accuracy],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)
            return cost, accuracy

        #Generate batches
        batches = dataManager.seq2seq_batch_iter(x, y_label, FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(len(x)/FLAGS.batch_size) + 1
        #Training loop. For each batch...
        num_batch = 1
        num_epoch = 1
        for batch in batches:
            if num_batch == num_batches_per_epoch:
                num_epoch += 1
                num_batch = 1
            num_batch += 1
            x_batch, y_label_b = zip(*batch)
            y_label_batch = one_hot(y_label_b)
            y_batch = decoder_y(x_batch)
            cost, accuracy = train_step(x_batch, y_batch, y_label_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("Num_batch: {}".format(num_batch))
                print("Num_epoch: {}".format(num_epoch))
                print("Cost:" + "{:.6f}".format(cost) + ", Accuracy:" + "{:.5f}".format(accuracy))
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

