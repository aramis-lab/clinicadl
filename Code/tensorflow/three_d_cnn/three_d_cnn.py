#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu March 08 09:51:28 2018

@author: junhao WEN

This is a three_d_cnn CNN version of lenet-5 architec (conv-pool-conv-pool-conv-pool-conv-pool-fullconnect-dropout-softmax)
"""



import numpy as np
np.random.seed(1234)
from three_d_cnn_utils import *
import os
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution3d, flatten, dropout
from tensorflow.python.layers.pooling import max_pooling3d
from tensorflow.python.ops.nn import relu,softmax
from tensorflow.python.framework.ops import reset_default_graph
from Code.visualization.tf_cnnvis.tf_cnnvis.tf_cnnvis import _save_model



def lenet_adopted_3dcnn(l_reshape, num_filters_conv1, num_filters_conv2, num_filters_conv3, num_filters_conv4,
                        num_fc1, num_classes, kernel_size_conv, conv_stride_size, pool_size, pool_stride_size, is_training, pool_padding='valid'):
    """
        This is an adopted lenet of three_d_cnn CNN.
    :param l_reshape: Tensor, with shape NHWC of tensorflow
    :param num_filters_conv1: int, number of filters used in this convnet
    :param num_filters_conv2:
    :param num_filters_conv3:
    :param num_filters_conv4:
    :param num_fc1: number of neurons used in the fully connected layer
    :param num_classes: int, numbers of classese in the dataset
    :param kernel_size_conv: the kenel size. e.g. e.g. [3,3,3]
    :param conv_stride_size: the stide size for conv. e.g. [1,1,1] # [stride_height, stride_width]
    :param pool_size: the pool size. e.g. [2,2,2]
    :param pool_stride_size: the pool stride size. e.g. [2,2,2]
    :param is_training: tensor, boolen, indicate if this is training or testing to decide if apply drop out.
    :return:
    """
    # Building the layers of the neural network
    # we define the variable scope, so we more easily can recognise our variables later
    l_conv1 = convolution3d(l_reshape, num_filters_conv1, kernel_size_conv, conv_stride_size, activation_fn=relu,
                            scope="l_conv1")

    l_maxpool1 = max_pooling3d(l_conv1, pool_size, pool_stride_size, padding=pool_padding)

    l_conv2 = convolution3d(l_maxpool1, num_filters_conv2, kernel_size_conv, conv_stride_size, activation_fn=relu,
                            scope="l_conv2")

    l_maxpool2 = max_pooling3d(l_conv2, pool_size, pool_stride_size, padding=pool_padding)

    l_conv3 = convolution3d(l_maxpool2, num_filters_conv3, kernel_size_conv, conv_stride_size, activation_fn=relu,
                            scope="l_conv3")

    l_maxpool3 = max_pooling3d(l_conv3, pool_size, pool_stride_size, padding=pool_padding)

    l_conv4 = convolution3d(l_maxpool3, num_filters_conv4, kernel_size_conv, conv_stride_size, activation_fn=relu,
                            scope="l_conv4")

    l_flatten = flatten(l_conv4, scope="flatten")  ## falten for fc layer

    l1 = fully_connected(l_flatten, num_fc1, activation_fn=relu, scope="l1")

    l_fc1 = dropout(l1, is_training=is_training, scope="dropout", keep_prob=0.1)

    y = fully_connected(l_fc1, num_classes, activation_fn=softmax, scope="y")

    return l_conv1, l_maxpool1, l_conv2, l_maxpool2, l_conv3, l_maxpool3, l_conv4, l_fc1, y

def train_adni_mri(caps_directory, subjects_visits_tsv, diagnoses_tsv, n_fold, batch_size, num_epochs, log_dir,
                  learning_rate=0.0001, num_classes=2, modality='t1'):
    """
    This is the main function to run the CD CNN with adni t1 image, including:
        -- prepare the data
        -- build the CNN
        -- training and validation
        -- testing with an isolated dataset
    :param caps_directory:
    :param subjects_visits_tsv:
    :param diagnoses_tsv:
    :param n_fold: using sklearn StratifiedKfold strategy to split the dataset into training, validationa and testing dataset
    :param num_classes: number of classes in your data, default is 2, a binary classification
    :param batch_size: the number of subjects in each batch
    :param num_epochs: how many epochs that you wanna trin in each fold
    :param log_dir: path to the logs
    :param learning_rate: the learning rate or initial leaning rate if you want to use exponential_decay strategy.
    :return:
    """
    
    ### check if the log folder exist
    check_and_clean(log_dir)    

    train_accuracy = np.zeros((n_fold,))
    test_accuracy = np.zeros((n_fold,))
    valid_accuracy = np.zeros((n_fold,))
    for fi in range(n_fold):

        print('Now running on fold %d'%(fi+1))

        if modality == 't1':
            #### prepare the dataset of ADNI_T1
            x_train, y_train, x_test, y_test, x_valid, y_valid, size_input = load_adni_mri(fi, n_fold, caps_directory, subjects_visits_tsv, diagnoses_tsv, image_type='T1', sizeX=121, sizeY=145, sizeZ=121)
        elif modality == 'dti':
            x_train, y_train, x_test, y_test, x_valid, y_valid, size_input = load_adni_mri(fi, n_fold, caps_directory, subjects_visits_tsv, diagnoses_tsv, image_type='dti', sizeX=182, sizeY=218, sizeZ=182)


        nchannels, depths, widths, lengths = size_input ## tf format is NDHWC
        x_train = x_train.astype('float32')
        x_train = x_train.reshape((-1, depths, widths, lengths, nchannels))
        targets_train = y_train.astype('int32')

        x_valid = x_valid.astype('float32')
        x_valid = x_valid.reshape((-1, depths, widths, lengths, nchannels))
        targets_valid = y_valid.astype('int32')

        x_test = x_test.astype('float32')
        x_test = x_test.reshape((-1, depths, widths, lengths, nchannels))
        targets_test = y_test.astype('int32')

        ### define the parameters of the CNN
        num_filters_conv1 = 5
        num_filters_conv2 = 5
        num_filters_conv3 = 5
        num_filters_conv4 = 5
        kernel_size_conv = [3, 3, 3] # [height, width]
        pool_size = [2, 2, 2]
        pool_stride_size = [2, 2, 2]
        conv_stride_size = [1, 1, 1] # [stride_height, stride_width]
        num_fc1 = 100 ## number of neuros for fc layer
        # resetting the graph ...
        reset_default_graph()

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, [None, depths, widths, lengths, nchannels])
        # l_reshape = tf.transpose(x_pl, [0, 2, 3, 4, 1]) # TensorFlow uses NHWC instead of NCHW
        y_ = tf.placeholder(tf.float32, [None, num_classes])
        is_training = tf.placeholder(tf.bool)#used for dropout

        ### Build the CNN
        l_conv1, l_maxpool1, l_conv2, l_maxpool2, l_conv3, l_maxpool3, l_conv4, l_fc1, y = lenet_adopted_3dcnn(x_pl, num_filters_conv1, num_filters_conv2, num_filters_conv3, num_filters_conv4,
                        num_fc1, num_classes, kernel_size_conv, conv_stride_size, pool_size, pool_stride_size, is_training)

        ### Train CNN
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
            loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Decay the learning rate exponentially based on the number of steps.
        global_step = tf.Variable(0, trainable=False)
        num_batches_per_epoch = x_train.shape[0] / batch_size
        NUM_EPOCHS_PER_DECAY = 10  # Epochs after which learning rate decays.
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.

        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        x_train.shape[0],
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        with tf.name_scope('train'):
            # defining our optimizer
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            # applying the gradients
            train_step = optimizer.minimize(cross_entropy)

        ### Run the CNN
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        writer_1 = tf.summary.FileWriter(os.path.join(log_dir, "log_dir"+"_fold"+str(fi), "train"), sess.graph)
        writer_2 = tf.summary.FileWriter(os.path.join(log_dir, "log_dir"+"_fold"+str(fi), "val"))
        writer_3 = tf.summary.FileWriter(os.path.join(log_dir, "log_dir"+"_fold"+str(fi), "test"))
        writer_1.add_graph(sess.graph)
        writer_2.add_graph(sess.graph)
        writer_3.add_graph(sess.graph)
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)

        ### check the activation map, just save one slice
        featuermaps = sess.run(l_conv1, feed_dict={
            x_pl: x_train[0].reshape(1, depths, widths, lengths, nchannels)})
        tensor_featuermaps = tf.reshape(tf.convert_to_tensor(featuermaps, np.float32)[:, :, :, :, 0][:, :, :, 60], (1, depths, widths, 1))
        tf.summary.image("feature_map1", tensor_featuermaps)

        write_op = tf.summary.merge_all()
        summary_pb = tf.summary.Summary()

        steps = int(x_train.shape[0] / batch_size)

        for i in range(num_epochs):
            train_acc = []
            valid_acc = []
            test_acc = []

            for ii in range(steps):
                # Calculate our current step
                step = i * steps + ii
                idx = range(ii * batch_size, (ii + 1) * batch_size)
                x_batch = x_train[idx]
                target_batch = targets_train[idx]
                # Feed forward batch of train images into graph and log accuracy
                feed_dict_train = {x_pl: x_batch, y_: onehot(target_batch, num_classes), is_training: True}

                if step % 5 == 0:
                    # Get Train Summary for one batch and add summary to TensorBoard
                    summary_train = sess.run([write_op, accuracy], feed_dict=feed_dict_train)
                    summary_pb.ParseFromString(summary_train[0])
                    writer_1.add_summary(summary_train[0], step)
                    writer_1.flush()
                    ## add the accuracy into the train_acc
                    train_acc.append(summary_train[1])

                    # Get Test Summary on random 10 test images and add summary to TensorBoard
                    x_valid, targets_valid = shuffle(x_valid, targets_valid)
                    x_batch = x_valid[0:10, :, :, :]
                    target_batch = targets_valid[0:10]
                    feed_dict_val = {x_pl: x_batch, y_: onehot(target_batch, num_classes), is_training: False}
                    summary_val = sess.run(write_op, feed_dict=feed_dict_val)
                    writer_2.add_summary(summary_val, step)
                    writer_2.flush()
                    acc_val = sess.run(accuracy, feed_dict=feed_dict_val)
                    valid_acc.append(acc_val)
                else:

                    acc_train = sess.run(accuracy, feed_dict=feed_dict_train)
                    train_acc.append(acc_train)

                # Back propigate using adam optimizer to update weights and biases.
                sess.run(train_step, feed_dict=feed_dict_train)

            print('Epoch number {} Training Accuracy: {}'.format(i + 1, np.mean(train_acc)))
	    print('Epoch number {} Validation Accuracy: {}'.format(i + 1, np.mean(valid_acc)))

        # Feed forward all test images into graph and log accuracy
        for iii in range(int(x_test.shape[0] / batch_size)):
            idx = range(iii * batch_size, (iii + 1) * batch_size)
            x_batch = x_test[idx]
            target_batch = targets_test[idx]
            feed_dict_test = {x_pl: x_batch, y_: onehot(target_batch, num_classes), is_training: False}
            acc_test = sess.run(accuracy, feed_dict=feed_dict_test)
            test_acc.append(acc_test)
            summary_test = sess.run(write_op, feed_dict=feed_dict_test)
            writer_3.add_summary(summary_test, iii)
            writer_3.flush()

        print("Test Set Accuracy: {}".format(np.mean(test_acc)))

        test_accuracy[fi] = np.mean(test_acc)

        ### save the model into google proto
        if fi == n_fold-1:
            model_path = _save_model(sess)
        sess.close()

    print("\n\n")
    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print '\nMean accuray of testing set: %f %%' % (np.mean(test_accuracy)*100)







