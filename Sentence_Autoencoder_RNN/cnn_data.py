#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from DataManager import DataManager
from RE_CNN import TextCNN
from tensorflow.contrib import learn

dataManager = DataManager()

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1481741016/checkpoints", "Checkpoint directory from training run")
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

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_test:
    x_text, _, _ = dataManager.load_training_data()
else:
    x_test = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

x_test = []
for data in x_text:
    a = 100-len(data)
    if a > 0:
        front = a/2
        back = a-front
        front_vec = [np.zeros(dataManager.wordvector_dim+2) for j in range(front)]
        back_vec = [np.zeros(dataManager.wordvector_dim+2) for k in range(back)]
        data = np.asarray(front_vec+data+back_vec)
    else:
        data = np.asarray(data[:100])
    x_test.append(data)
x_test = np.asarray(x_test)

output_file = open("data/relation_feature", "w")

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
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        hidden_feature = graph.get_operation_by_name("hidden_feature").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = dataManager.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            relation_feature = sess.run(hidden_feature, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            for f in relation_feature:
                np.save(output_file, f)

