# coding: utf8

"""This is the file to handle the ADNI png images, basically, the png file is gray scale image saved as png format and the width is 145, the length is 121.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

from Code import adni_png_main

author = "Junhao WEN"
copyright = "Copyright 2016-2018 The Aramis Lab Team"
credits = ["Junhao WEN"]
license = "See LICENSE.txt file"
version = "0.1.0"
maintainer = "Junhao WEN"
email = "junhao.wen@inria.fr"
status = "Development"

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """Number of classes in the dataset, by default, it is a binary classification.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/DL_AD/Data',
                           """Path to the ADNI png binary files data directory.""")
tf.app.flags.DEFINE_string('tensorboard_log_dir',
                           '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/DL_AD/Results',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('train_binary', 'training_adni_png_AD_vs_CN_baseline_T1_slices_32684.bin',
                           """binary file for training dataset""")
tf.app.flags.DEFINE_string('test_binary', 'test_adni_png_AD_vs_CN_baseline_T1_slices_8249.bin',
                           """binary file for test dataset""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of steps to run, each step has random samples with batch_size.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_float('dropout_rate', 0.5,
                          """the probalility for dropout, if set to 1, means no dropout for validation or test.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 1,
                            """How many epoch to update the decayed learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.96,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 62881,
                            """How many pngs you want to train, by default, it is the num of pngs for training dataset""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_test', 15976,
                            """How many pngs you want to train, by default, it is the num of pngs for test dataset""")
tf.app.flags.DEFINE_integer('png_width', 145,
                            """original image  in width""")
tf.app.flags.DEFINE_integer('png_length', 121,
                            """original image  in length""")
tf.app.flags.DEFINE_integer('png_depth', 1,
                            """original image  in depth""")
tf.app.flags.DEFINE_integer('image_width_rand_crop', 145,
                            """resize the original image into a specific size in width""")
tf.app.flags.DEFINE_integer('image_length_rand_crop', 121,
                            """resize the original image into a specific size in length""")
tf.app.flags.DEFINE_boolean('test_eval_once', True,
                            """Whether to run eval only once.""")

def train():
  """Train ADNI png for a number of steps."""

  if tf.gfile.Exists(FLAGS.tensorboard_log_dir + '/trainlog'):
    tf.gfile.DeleteRecursively(FLAGS.tensorboard_log_dir + '/trainlog')
  tf.gfile.MakeDirs(FLAGS.tensorboard_log_dir + '/trainlog')

  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for ADNI png.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images_training, labels_training = adni_png_main.inputs(FLAGS.train_binary)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    Y_logits, conv1, conv2 = adni_png_main.adopted_adni_png_net(images_training, dropout_rate=FLAGS.dropout_rate)

    # Calculate loss.
    loss_training = adni_png_main.loss(Y_logits, labels_training)

    with tf.name_scope('accuracy_training'):
        top_k_op_training = tf.nn.in_top_k(Y_logits, labels_training, 1)
        accuracy_training = tf.reduce_mean(tf.cast(top_k_op_training, tf.float32))

    tf.summary.scalar(accuracy_training.op.name + ' train_training', accuracy_training)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = adni_png_main.train(loss_training, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss_training, accuracy_training])  # Asks for loss and accuracy value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results[0]
          accuracy_value = run_values.results[1]

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f, accurcy = %.2f, (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value, accuracy_value,
                               examples_per_sec, sec_per_batch))

    ## For training.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.tensorboard_log_dir + '/trainlog',
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss_training),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement), save_summaries_steps=5) as mon_sess:
      while not mon_sess.should_stop():
        ### TODO add validation process
        mon_sess.run(train_op)

# def main(argv=None):  # pylint: disable=unused-argument
#   if tf.gfile.Exists(FLAGS.tensorboard_log_dir + '/trainlog'):
#     tf.gfile.DeleteRecursively(FLAGS.tensorboard_log_dir + '/trainlog')
#   tf.gfile.MakeDirs(FLAGS.tensorboard_log_dir + '/trainlog')
#   train()
#
# if __name__ == '__main__':
#   tf.app.run()
