# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from Code import adni_png_main
from Code.adni_png_train import FLAGS

author = "Junhao WEN"
copyright = "Copyright 2016-2018 The Aramis Lab Team"
credits = ["Junhao WEN"]
license = "See LICENSE.txt file"
version = "0.1.0"
maintainer = "Junhao WEN"
email = "junhao.wen@inria.fr"
status = "Development"

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.tensorboard_log_dir + '/trainlog')
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples_per_epoch_for_test / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      # total_sample_count = num_iter * FLAGS.batch_size
      total_sample_count = FLAGS.num_examples_per_epoch_for_test
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval ADNI png for a number of steps."""

  if tf.gfile.Exists(FLAGS.tensorboard_log_dir + '/testlog'):
    tf.gfile.DeleteRecursively(FLAGS.tensorboard_log_dir + '/testlog')
  tf.gfile.MakeDirs(FLAGS.tensorboard_log_dir + '/testlog')

  with tf.Graph().as_default() as g:
    # Get images and labels for test dataset
    images, labels = adni_png_main.inputs(FLAGS.test_binary)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    Y_logits, conv1, conv2 = adni_png_main.adopted_adni_png_net(images, dropout_rate=1)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(Y_logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.tensorboard_log_dir + '/testlog', g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.test_eval_once:
        break
      time.sleep(10)

