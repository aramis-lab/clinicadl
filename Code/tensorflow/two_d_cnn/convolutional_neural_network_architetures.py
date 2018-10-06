#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Junhao WEN"
__copyright__ = "Copyright 2016, The Aramis Lab Team"
__credits__ = ["Junhao WEN"]
__license__ = "??"
__version__ = "1.0.0"
__maintainer__ = "Junhao WEN"
__email__ = "junhao.wen@inria.fr"
__status__ = "Development"


###############NOTE: as the DNN architecture was implemented with tensorflow virtualenv, try source this virtualenv

from tensorflow.contrib.learn import DNNClassifier
import tensorflow as tf
import numpy as np
import tempfile
from Code.two_d_cnn.classification_utils import _variable_with_weight_decay, _variable_on_cpu, _activation_summary

#############################################################################
############################ Linear classfiers                             #
#############################################################################
def TensorFlow_DNN(training_x, training_y, testing_x, testing_y, hidden_units, n_classes=2,
                   activation_fn=tf.nn.relu, dropout=None, working_directory=None, optimizer=None, steps_training=2000):
    """
    This is a function to use TF DNNClassifier to run classification, by default, it will be a binary classification
    :param traing_X: an array, size: n_sample*n_features
    :param traing_y: an array, size: n_sample*1
    :param testing_x:
    :param testing_y:
    :param hidden_units: list, hidden units per layer which are fully
    :param feature_columns: an iterable containing all the feature columns used by the model from TF FeatureColumn class
    :param n_classes: number of calsses of this classification, by default, n_classes is 2: a binary classification
    :param activation_fn: activaion funcation used for each layer, by default is relu, could also be tanh and sigmoid
    :param dropout: Boolen, if use dropout, when not 0, the probalility to use for dropout
    :param working_directory: the directory to contain the model parameters, graph and tec, this could be used to load
	checkpoints from the directory into a estimator to continue training a previously saved model
    :param optimizer: the method used to optimizer the training step, if none, Adagrad will be used
    :return:
    """
    if working_directory is None:
        working_directory = tempfile.mktemp()

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=training_x.shape[1])]

    # Evaluating the classifier for testing dataset
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        testing_x, testing_y, every_n_steps=50
    )


    # Build 3 layer DNN with several units that you define.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=hidden_units,
                                                n_classes=n_classes,
                                                model_dir=working_directory,
                                                dropout=dropout,
                                                activation_fn=activation_fn,
                                                optimizer=optimizer,
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=100))

    # Fit model.
    classifier.fit(x=training_x,
                   y=training_y,
                   steps=steps_training,
                   monitors=[validation_monitor])

    # Evaluate accuracy.
    accuracy_score_training = classifier.evaluate(x=training_x,
                                         y=training_y)["accuracy"]
    print('Accuracy for training dataset: {0:f}'.format(accuracy_score_training))

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=testing_x,
                                         y=testing_y)["accuracy"]
    print('Accuracy for testing dataset: {0:f}'.format(accuracy_score))


def alexnet_adpoted(images, batch_size, n_classes, weight_decay):
  """Build the adopted alexNet model.
    paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

  Args:
    images: a tf tensor, placeholder to hold the training dataset.

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=weight_decay)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=weight_decay)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=weight_decay)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=weight_decay)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entroplabel_tfwith_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, n_classes],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [n_classes],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear, conv1, conv2

def lenet_adopted(data, num_channel, n_classes, dropout_rate, weight_decay):
    """The Model definition."""
    with tf.variable_scope('conv1') as scope:
        conv1_weights = _variable_with_weight_decay('weight1',
                                             shape=[5, 5, num_channel, 32],
                                             stddev=5e-2,
                                             wd=weight_decay)
        conv1_biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32),
                                 trainable=True)

        conv1 = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu1 = tf.nn.leaky_relu(tf.nn.bias_add(conv1, conv1_biases), name=scope.name)

        _activation_summary(relu1)

    # POOL 1
    with tf.variable_scope('pool1') as scope:
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu1,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name)

    # NORM 1
    with tf.variable_scope('norm1') as scope:
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name = scope.name)

    with tf.variable_scope('conv2') as scope:
        conv2_weights = _variable_with_weight_decay('weight2',
                                             shape=[5, 5, 32, 64],
                                             stddev=5e-2,
                                             wd=weight_decay)

        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32),
                                 trainable=True)

        conv2 = tf.nn.conv2d(norm1,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.leaky_relu(tf.nn.bias_add(conv2, conv2_biases), name=scope.name)
        _activation_summary(relu2)

    # NORM 2
    with tf.variable_scope('norm2') as scope:
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope.name)

    # POOL 2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(norm2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name)


    #FULLY CONNECTED 1
    with tf.variable_scope('fc1') as scope:
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        shape = int(np.prod(pool2.get_shape()[1:]))
        fc1_weights = tf.Variable(tf.truncated_normal([shape, 512], dtype=tf.float32,
                                               stddev=1e-1))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden1 = tf.nn.leaky_relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
	if dropout_rate == 1:
	    pass
        else:
	    hidden1 = tf.nn.dropout(hidden1, dropout_rate, name=scope.name)
        _activation_summary(hidden1)

    #FULLY CONNECTED 3 & SOFTMAX OUTPUT
    with tf.variable_scope('fc2_softmax') as scope:
        fc2_weights = tf.Variable(tf.truncated_normal([512, n_classes], dtype=tf.float32,
                                               stddev=1e-1))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[n_classes], dtype=tf.float32))

        Y_logits = tf.add(tf.matmul(hidden1, fc2_weights), fc2_biases, name=scope.name)

        _activation_summary(Y_logits)

        Y = tf.nn.softmax(Y_logits)

    return Y_logits, Y


