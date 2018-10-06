#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## for orignial image, it 145*121

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/network/lustre/iss01/home/junhao.wen/Project/AD-DL/Data/subject_level',
                           """Path to the ADNI T1 2D png data directory.""")
tf.app.flags.DEFINE_string('log_dir', '/network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/Lenet_log',
                           """Path to log dir for tensorboard usage.""")
tf.app.flags.DEFINE_integer('image_width', 145,
                            """the png image size, width.""")
tf.app.flags.DEFINE_integer('image_length', 121,
                            """the png image size, length.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """numbers of channels of image, 1 is grayscale, 3 is rgb, 4 is rgba.""")
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """inter, how many classes""")
tf.app.flags.DEFINE_integer('num_epochs', 150,
                            """the number of epochs to run.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                            """the initial learning rate for the network.""")
tf.app.flags.DEFINE_float('dropout_rate', 0.5,
                          """the probalility for dropout, if set to 1, means no dropout for validation or test.""")
tf.app.flags.DEFINE_float('weight_decay', 0,
                          """the weight decay used in the network for regularization""")
#tf.app.flags.DEFINE_string('training_bin', 'training_adni_AD_vs_CN_baseline_T1_slices_55233.bin',
 #                          """binary file for training dataset""")
#tf.app.flags.DEFINE_string('test_bin', 'test_adni_AD_vs_CN_baseline_T1_slices_14008.bin',
 #                          """binary file for test dataset""")
#tf.app.flags.DEFINE_string('validation_bin', 'test_adni_AD_vs_CN_baseline_T1_slices_14008.bin',
 #                          """binary file for validation dataset""")
tf.app.flags.DEFINE_integer('n_fold', 5,
                            """Which fold you are runing""")


