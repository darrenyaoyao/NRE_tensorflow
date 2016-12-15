#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from DataManager import DataManager
from seq2seq_model import Seq2seqModel

dataManager = DataManager()

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("sentence_length", 80, "Sentence length (default: 80)")
tf.flags.DEFINE_string("checkpoint_dir", "./seq2seq_runs/1481712181/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_test", True, "Evaluate on all testing data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading testing data...")
x_text, _, y_num = dataManager.load_training_data()
print("Finish loading data")

# Prepare data
x = []
for data in x_text:
    a = FLAGS.sentence_length-len(data)
    if a > 0:
        padding = [np.zeros(52) for i in range(a)]
        x.append(data+padding)
    else:
        x.append(data[:FLAGS.sentence_length])

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
        a = FLAGS.sentence_length-len(s)
        if a > 0:
            padding = [0 for k in range(a)]
            y[i] = y[i]+padding
        else:
            y[i] = y[i][:FLAGS.sentence_length]
        for j in range(FLAGS.sentence_length):
            v = np.zeros(8000, dtype=np.float32)
            v[y[i][j]] = 1.0
            y[i][j] = v
        y_l.append(np.asarray(y[i], dtype=np.float32))
    return np.asarray(y_l, dtype=np.float32)

y_label = y_num

output_file = open("data/sentence_feature", "w")

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        output_y = graph.get_operation_by_name("output_y").outputs[0]
        hidden_state = graph.get_operation_by_name("hidden_state").outputs[0]

        # Generate batches for one epoch
        batches = dataManager.seq2seq_batch_iter(x, y_label, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for batch in batches:
            x_batch, y_label_b = zip(*batch)
            y_label_batch = one_hot(y_label_b)
            y_batch = decoder_y(x_batch)
            sentence_feature = sess.run(hidden_state, {input_x: x_batch,
                                                        input_y: y_batch,
                                                        output_y: y_label_batch})
            for f in sentence_feature[0]:
                np.save(output_file, f)
'''
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
'''
