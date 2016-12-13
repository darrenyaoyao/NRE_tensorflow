import tensorflow as tf
import numpy as np
import os
import time
import datetime
from DataManager import DataManager
from Sentence_Autoencoder import SenAutoencoder

# Parameters
# =======================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionally of character embedding.")
tf.flags.DEFINE_integer("filter_sizes", 3, "Comma-separated filter sizes (default: 3)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
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

dataManager = DataManager()

#Load data
print("Loading training data...")
x_text, y = dataManager.load_training_data()
print("Finish loading data")

x = []
for data in x_text:
    a = 100-len(data)
    if a > 0:
        front = a/2
        back = a - front
        front_vec = [np.zeros(dataManager.wordvector_dim + 2) for j in range(front)]
        back_vec = [np.zeros(dataManager.wordvector_dim + 2) for k in range(back)]
        data = np.asarray(front_vec+data+back_vec)
    else:
        data = np.asarray(data[:100])
    x.append(data)
x = np.asarray(x)

#Random shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x[shuffle_indices]
y_train = y[shuffle_indices]

# Training
# ==============================================

print("Start Training")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        senautoencoder = SenAutoencoder(
            sequence_length = x_train.shape[1],
            embedding_size=dataManager.wordvector_dim+2,
            filter_sizes = FLAGS.filter_sizes,
            num_filters = FLAGS.num_filters,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(senautoencoder.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "autoencoder_runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", senautoencoder.loss)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              senautoencoder.input_x: x_batch,
              senautoencoder.input_y: x_batch,
            }
            _, step, summaries, loss= sess.run(
                [train_op, global_step, train_summary_op, senautoencoder.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)
            return loss

        # Generate batches
        batches = dataManager.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(len(x_train)/FLAGS.batch_size) + 1
        # Training loop. For each batch...
        num_batch = 1
        num_epoch = 1
        for batch in batches:
            if num_batch == num_batches_per_epoch:
                num_epoch += 1
                num_batch = 1
            num_batch += 1
            x_batch, y_batch = zip(*batch)
            loss = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("Num_batch: {}".format(num_batch))
                print("Num_epoch: {}".format(num_epoch))
                print("Loss: {}".format(loss))
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

