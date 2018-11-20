#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Simple, end-to-end, LeNet-5-like CNN on ADNI 2D png images.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from Code.tensorflow.two_d_cnn.classification_utils import *
from Code.tensorflow.two_d_cnn.convolutional_neural_network_architetures import lenet_adopted
import tensorflow as tf
from Code.tensorflow.two_d_cnn.flags_slice_mixed import FLAGS
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def adni_two_d_lenet():
    """
    Training the adni with 2D cnn in slice-level
    :return:
    """
    check_and_clean(FLAGS.log_dir)
    test_accuracy = np.zeros((FLAGS.n_fold,))

    ## read the fold 0 from the subject-level data, and resplit it into 5-fold with StratifiedKFold
    training_bin = "training_adni_AD_vs_CN_baseline_T1_fold_" + str(0) + ".bin"
    validation_bin = "validation_adni_AD_vs_CN_baseline_T1_fold_" + str(0) + ".bin"
    test_bin = "test_adni_AD_vs_CN_baseline_T1_fold_" + str(0) + ".bin"

    x_train, y_train, size_input = shuffle_adni(FLAGS.data_dir, training_bin)
    x_valid, y_valid, size_input = shuffle_adni(FLAGS.data_dir, validation_bin)
    x_test, y_test, size_input = shuffle_adni(FLAGS.data_dir, test_bin)

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
    train_id = [''] * FLAGS.n_fold
    test_id = [''] * FLAGS.n_fold
    a = 0

    for train_index, test_index in skf.split(np.concatenate((x_train, x_valid, x_test)), np.concatenate((y_train, y_valid, y_test))):
        train_id[a] = train_index
        test_id[a] = test_index
        a = a + 1


    for fi in range(FLAGS.n_fold):
	
	test_true_label = []
	test_predicted_label = []
	
        testid = test_id[fi]
        trainid = train_id[fi]
        x_train = np.concatenate((x_train, x_valid, x_test))[trainid]
        y_train = np.concatenate((y_train, y_valid, y_test))[trainid]

        skf_2 = StratifiedKFold(2, shuffle=False, random_state=0)
        for test_ind, valid_ind in skf_2.split(np.concatenate((x_train, x_valid, x_test))[testid], np.concatenate((y_train, y_valid, y_test))[testid]):
            print("SPLIT iteration:", "Test:", test_ind, "Validation", valid_ind)

        x_valid = np.concatenate((x_train, x_valid, x_test))[testid][valid_ind]
        y_valid = np.concatenate((y_train, y_valid, y_test))[testid][valid_ind]
        x_test = np.concatenate((x_train, x_valid, x_test))[testid][test_ind]
        y_test = np.concatenate((y_train, y_valid, y_test))[testid][test_ind]

        # Get the data.
	#x_train = zero_center_dataset(data_train, data_train)
        #x_test = zero_center_dataset(data_test, data_train)
        #x_valid = zero_center_dataset(data_valid, data_train)

        # normalize the intensity value to [0, 1]
        #x_train = tf.keras.utils.normalize(zero_center_dataset(data_train, data_train), axis=0)
        #x_test = tf.keras.utils.normalize(zero_center_dataset(data_test, data_train), axis=0)
        #x_valid = tf.keras.utils.normalize(zero_center_dataset(data_valid, data_train), axis=0)
	
	### respl
	
        # As a sanity check, print out the shapes of the data
        print('Training data shape: %s' %  str(x_train.shape))
        print('Test data shape: %s' % str(x_test.shape))
        print('Validation data shape: %s' % str(x_valid.shape))
	
	###  reset the graph for a new fold
	tf.reset_default_graph() 
        
	image_tf = tf.placeholder(
          tf.float32,
          shape=(FLAGS.batch_size, FLAGS.image_width, FLAGS.image_length, FLAGS.num_channels))
        label_tf = tf.placeholder(tf.int8, [None, FLAGS.num_classes])

        dropout_rate = tf.placeholder(tf.float32)

        # Predictions for the test and validation, which we'll compute less often.
        y_logist, y_sofmax = lenet_adopted(image_tf, 1, 2, dropout_rate, FLAGS.weight_decay)


        with tf.name_scope('cross_entropy'):
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logist, labels=label_tf)
          loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
          predicted_label = tf.argmax(y_sofmax, 1)  
	  correct_prediction = tf.equal(tf.argmax(y_sofmax, 1), tf.argmax(label_tf, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Decay the learning rate exponentially based on the number of steps.
        global_step = tf.Variable(0, trainable=False)
        LEARNING_RATE_DECAY_FACTOR = 0.995  # Learning rate decay factor.

        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      x_train.shape[0],
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
        tf.summary.scalar('learning_rate', lr)

        with tf.name_scope('train'):
          train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
          # train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

        ### creat a session to start feeding the data
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        writer_1 = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "log_dir" + "_fold" + str(fi), "train"))
        writer_2 = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "log_dir" + "_fold" + str(fi), "test"))
        writer_3 = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "log_dir" + "_fold" + str(fi), "val"))

        writer_1.add_graph(sess.graph)
        writer_2.add_graph(sess.graph)
	writer_3.add_graph(sess.graph)

        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.image('Image', x_train[0:1, :, :, :])
        write_op = tf.summary.merge_all()

        steps_train = int(x_train.shape[0] / FLAGS.batch_size)
	steps_val = int(x_valid.shape[0] / FLAGS.batch_size)
	step_val = -1

        for i in range(FLAGS.num_epochs):
          train_acc = []
          test_acc = []

          for ii in range(steps_train):
              # Calculate our current step
              step_train = i * steps_train + ii
              idx = range(ii * FLAGS.batch_size, (ii + 1) * FLAGS.batch_size)
              x_batch_train = x_train[idx]
              target_batch_train = y_train[idx]
              print('For step %d, there are %d CN in this batch' % (step_train, int(target_batch_train.sum())))
              # Feed forward batch of train images into graph and log accuracy
              feed_dict_train = {image_tf: x_batch_train, label_tf: onehot(target_batch_train, FLAGS.num_classes), dropout_rate: FLAGS.dropout_rate}

              if step_train % 10 == 0:
		  step_val += 1 
		  if step_val == steps_val:
		      step_val = 1
		  idx_val = range(step_val * FLAGS.batch_size, (step_val + 1) * FLAGS.batch_size)
		      		
                  # Get Train Summary for one batch and add summary to TensorBoard
                  summary_train = sess.run([write_op, accuracy, loss, predicted_label], feed_dict=feed_dict_train)
                  writer_1.add_summary(summary_train[0], step_train)
                  writer_1.flush()
                  ## add the accuracy into the train_acc
                  train_acc.append(summary_train[1])
                  print('For step %d, the training accuracy is %f' % (step_train, summary_train[1]))
                  print('For step %d, the training loss is %f' % (step_train, summary_train[2]))
                  print('For step %d, the true label are %s' % (step_train, str(target_batch_train)))
                  print('For step %d, the predicted label are %s' % (step_train, str(summary_train[3])))
                  x_batch_val = x_valid[idx_val]
                  target_batch_val = y_valid[idx_val]
		  feed_dict_val = {image_tf: x_batch_val, label_tf: onehot(target_batch_val, FLAGS.num_classes), dropout_rate: 1}
                  summary_val = sess.run([write_op, accuracy, loss, predicted_label], feed_dict=feed_dict_val)
                  writer_3.add_summary(summary_val[0], step_train)
                  writer_3.flush()
                  print("For validation step %d, the true label are %s " % (step_train, str(target_batch_val)))
                  print('For validation step %d, there are %d CN in this batch' % (step_train, int(target_batch_val.sum())))
                  print('For validation step %d, the predicted label are %s' % (step_train, str(summary_val[3])))
		  print('For validation step %d, the accuracy is %f' % (step_train, summary_val[1]))
		  print('For validation step %d, the loss is %f' % (step_train, summary_val[2]))	
              else:

                  acc_train = sess.run(accuracy, feed_dict=feed_dict_train)
                  train_acc.append(acc_train)

              # Back propigate using adam optimizer to update weights and biases.
              sess.run(train_step, feed_dict=feed_dict_train)
	  ### shuffle each training dataset after each epoch, ensure that we get a fairly good estimate of the true gradient.
	  x_train, y_train = shuffle(x_train, y_train)
	  x_valid, y_valid = shuffle(x_valid, y_valid)
          print('Epoch number {}: The mean training Accuracy: {}'.format(i + 1, np.mean(train_acc)))

        # Feed forward all test images into graph and log accuracy
        for iii in range(int(x_test.shape[0] / FLAGS.batch_size)):
          idx_test = range(iii * FLAGS.batch_size, (iii + 1) * FLAGS.batch_size)
          x_batch_test = x_test[idx_test]
          target_batch_test = y_test[idx_test]
          feed_dict_test = {image_tf: x_batch_test, label_tf: onehot(target_batch_test, FLAGS.num_classes), dropout_rate: 1}
          summary_test = sess.run([write_op, accuracy, predicted_label], feed_dict=feed_dict_test)
          test_acc.append(summary_test[1])
	  print("For testing step %d, the true label are %s " % (iii, str(target_batch_test)))
          print('For testing step %d, there are %d CN in this batch' % (iii, int(target_batch_test.sum())))
          print('For testing step %d, the predicted label are %s' % (iii, str(summary_test[2])))
          print('For testing step %d, the accuracy is %f' % (iii, summary_test[1]))
          writer_2.add_summary(summary_test[0], iii)
          writer_2.flush()
	  ### adding ture lable and the predicted label into a tsv file
	  test_true_label.append(target_batch_test.tolist())
	  test_predicted_label.append(summary_test[2].tolist())
	
        ### save in a tsv file
        df_true = pd.DataFrame(np.asarray(test_true_label).flatten(), columns=['true_label'])
	df_predicted = pd.DataFrame(np.asarray(test_predicted_label).flatten(), columns=['predicted_label'])
	df_all = df_true.join(df_predicted)
        df_all.to_csv(os.path.join(FLAGS.log_dir, "log_dir" + "_fold" + str(fi), 'test_results_slice_level' + '_fold_' +  str(fi) + '.tsv'),
                  index=False, sep='\t', encoding='utf-8')	  

        print("Mean test Accuracy is: {}".format(np.mean(test_acc)))
	test_accuracy[fi] = np.mean(test_acc)

        ### save the model into google proto
        model_path = save_model(sess, fi, FLAGS.log_dir)
        sess.close()

    print("\n\n")
    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print('\nMean accuray of testing set: %f' % (np.mean(test_accuracy) * 100))

