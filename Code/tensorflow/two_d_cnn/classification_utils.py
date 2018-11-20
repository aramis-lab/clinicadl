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

import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from Code.tensorflow.two_d_cnn.machine_learning_architectures import KNearestNeighbor, LinearSVM
from random import randrange
from math import sqrt, ceil
from scipy.ndimage import uniform_filter
import math, re
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import imshow
from array import array
configProto = tf.ConfigProto(allow_soft_placement = True)


def load_data_lables(filename, binary=False):
  """ load single batch of cifar """

  width = 145
  depth = 121
  with open(filename, 'rb') as f:
    if binary  == False: ### if the data is stored use pickle as a dictionary
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        #participant_id = datadict['participant_id']
    else: ## if the data is stored as binary
        data = f.read()
        data_array = np.frombuffer(data, dtype='<f4')
        print 'Shape:', data_array.shape
        num_pngs = data_array.shape[0] / (width * depth + 1)
        per_png_lable_img = width * depth * 1 + 1
        try:
            isinstance(num_pngs, int)
        except TypeError:
            raise
        X = np.zeros((num_pngs, width, depth, 1))
        Y = np.zeros((num_pngs,))
        #convert the one-dimension array back into label and image format
        for n in xrange(num_pngs):
            Y[n] = int(data_array[per_png_lable_img * n])
            X[n,:,:,:] = data_array[per_png_lable_img * n + 1: per_png_lable_img * (n + 1)].reshape((width, depth, 1))

	print('The label looks like this:\n')
	print(np.array2string(Y))

        ## plot one image as example for sanity check
        index_img = np.random.randint(0, num_pngs)
        #imshow(np.reshape(X[index_img,:], (145, 121)))

    ### The information for training, validation and test dataset.
    print('This dataset: \n')
    print('In total, we have %d pngs\n' % Y.shape[0])
    print('In total, we have %d CN\n' % int(Y.sum()))

    return X, Y

def shuffle_adni_png(ROOT, training_bin, testing_bin, fold_index, n_fold):
  """ load all of ADNI CN and AD data
      This function is used to split the whole dataset into training and testing dataset based on StratifiedKFold.
  """
  from sklearn.utils import shuffle
  import gc

  Xtr, Ytr = load_data_lables(os.path.join(ROOT, training_bin), binary=True)
  Xte, Yte = load_data_lables(os.path.join(ROOT, testing_bin), binary=True)

  A_num, width, length, num_channels = Xtr.shape
  B_num, _, _, _ = Xte.shape
  size_input = [1, width, length, num_channels]
  all_data = np.concatenate((Xtr, Xte), axis=0)
  all_label = np.concatenate((Ytr, Yte), axis=0)
  ## shuffle the image and label
  all_data, all_label = shuffle(all_data, all_label)
  print('Training + testing dataset: \n')
  print('In total, we have %d pngs\n' % all_label.shape[0])
  print('In total, we have %d CN\n' % int(all_label.sum()))

  skf = StratifiedKFold(n_fold)
  train_id = [''] * n_fold
  test_id = [''] * n_fold
  a = 0

  for train_index, test_index in skf.split(all_data, all_label):
      print("SPLIT iteration:", a + 1, "Traing:", train_index, "Test", test_index)
      train_id[a] = train_index
      test_id[a] = test_index
      a = a + 1

  testid = test_id[fold_index]
  trainid = train_id[fold_index]
  x_train = all_data[trainid]
  y_train = all_label[trainid]
  
  x_test_val = all_data[testid].reset_index(drop=True)
  y_test_val = all_label[testid].reset_index(drop=True)
	  
  skf_2 = StratifiedKFold(2)
  for test_ind, valid_ind in skf_2.split(x_test_val, y_test_val):
      print("SPLIT iteration:", "Test:", test_ind, "Validation", valid_ind)

  x_valid = x_test_val[valid_ind]
  y_valid = y_test_val[valid_ind]
  x_test = x_test_val[test_ind]
  y_test = y_test_val[test_ind]

  ### The information for training, validation and test dataset.
  print('For fold %d \n' % fold_index)
  print('Training dataset: \n')
  print('In total, we have %d pngs\n' % x_train.shape[0])
  print('In total, we have %d CN\n' % int(y_train.sum()))
  x_train, y_train = shuffle(x_train, y_train)
  print('test dataset: \n')
  print('In total, we have %d pngs\n' % x_test.shape[0])
  print('In total, we have %d CN\n' % int(y_test.sum()))
  x_test, y_test = shuffle(x_test, y_test)
  print('Validation dataset: \n')
  print('In total, we have %d subjects\n' % x_valid.shape[0])
  print('In total, we have %d CN\n' % int(y_valid.sum()))
  x_valid, y_valid = shuffle(x_valid, y_valid)
  return x_train, y_train, x_test, y_test, x_valid, y_valid, size_input

def shuffle_adni(ROOT, data_file):
  """ load all of ADNI CN and AD data
      This function is used to split the whole dataset into training and testing dataset based on StratifiedKFold.
  """
  from sklearn.utils import shuffle
  import gc

  X, y = load_data_lables(os.path.join(ROOT, data_file), binary=True)

  A_num, width, length, num_channels = X.shape
  size_input = [1, width, length, num_channels]
  ## shuffle the image and label
  X, y = shuffle(X, y)
  print('After shuffling the dataset: \n')
  print('In total, we have %d pngs\n' % y.shape[0])
  print('In total, we have %d CN\n' % int(y.sum()))

  return X, y, size_input

def zero_center_dataset(X, X_whole):
    """
    This is a function to normalize the image: Preprocessing: subtract the mean image of the whole dataset
    :param X:
    :return:
    """
    mean_image = np.mean(X_whole, axis=0)
    X -= mean_image.astype('uint8')
    return X

def plot_weight_svm(svm_ob, weight_width, weight_height, label_groups, color_channel=1):
    """
    This is a function to plot the weight map after doing the calssification with sklearn LinearSVM object
    :param svm_ob:sklearn LinearSVM object
    :param weight_width: image dimension 1
    :param weight_height:image dimension 2
    :param label_groups:list, concaining the labels for different groups
    :param color_channel: int, 1 for grayscale image, 3 for RGB image.
    :return:
    """
    w = svm_ob.W[:-1, :]  # strip out the bias
    w = w.reshape(weight_width, weight_height, color_channel, len(label_groups))
    w_min, w_max = np.min(w), np.max(w)
    classes = label_groups
    for i in xrange(len(label_groups)):
        plt.subplot(1, len(label_groups), i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

def preprocessing_pickled_data(pickled_data, num_training=50000, num_validation=14184, num_test=14351):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    X_train, y_train, X_test, y_test = shuffle_adni_png(pickled_data)

    # subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, replace=False)

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0).astype('uint8')
    X_train -= mean_image.astype('uint8')
    X_val -= mean_image.astype('uint8')
    X_test -= mean_image.astype('uint8')

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test

# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

def display_exa_img(classes, samples_per_class, X, y):
  """
  This is a func to display some example imgs in the dataset for each class with random samples_per_class images.
  :param classes:
  :param samples_per_class:
  :return:
  """

  num_classes = len(classes)
  for j, cls in enumerate(classes):
     idxs = np.flatnonzero(y == j) #Return indices that are non-zero in the flattened version of a
     idxs = np.random.choice(idxs, samples_per_class, replace=False)# this will randomly choose 10 images' index to display
     for i, idx in enumerate(idxs):
         plt_idx = i * num_classes + j + 1
         plt.subplot(samples_per_class, num_classes, plt_idx)
         plt.imshow(X[idx].astype('uint8'))# this is the real step to show the images
         plt.axis('off')
         if i == 0:
             plt.title(cls)
  plt.show()

def sub_ADNI(num_training, num_test, X_train, y_train, X_test, y_test):
  """
  This
  :param num_training:
  :param num_test:
  :return:
  """
  num_training = random.sample(range(0, 64184), num_training)
  X_train = X_train[num_training]
  y_train = y_train[num_training]

  num_test = random.sample(range(0, 14351), num_test)
  X_test = X_test[num_test]
  y_test = y_test[num_test]
  return X_train, y_train, X_test, y_test

def cross_validation_KNearestNeighbor(X_train, y_train, num_folds, k_choices):
    """
    This func is to do the cross validation for the training dataset of KNearestNeighbor
    :param X_train:
    :param y_train:
    :param num_folds:
    :param k_choices:
    :return:
    """

    ##### np.array_split splits array into multiple sub-arrays.
    X_train_folds = np.array(np.array_split(X_train, num_folds))
    y_train_folds = np.array(np.array_split(y_train, num_folds))
    print X_train_folds.shape, y_train_folds.shape

    k_to_accuracies = {}
    # run every possible k-nearest-neighbor classifier
    for k in k_choices:
        for n in xrange(num_folds):
            combinat = [x for x in xrange(num_folds) if x != n]
            x_training_dat = np.concatenate(X_train_folds[combinat])
            y_training_dat = np.concatenate(y_train_folds[combinat])
            classifier_k = KNearestNeighbor()
            classifier_k.train(x_training_dat, y_training_dat)
            y_cross_validation_pred = classifier_k.predict_labels(X_train_folds[n], k)
            num_correct = np.sum(y_cross_validation_pred == y_train_folds[n])
            accuracy = float(num_correct) / y_train_folds[n].shape[0]
            k_to_accuracies.setdefault(k, []).append(accuracy)

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print 'k = %d, accuracy = %f' % (k, accuracy)
        print 'mean for k=%d is %f' % (k, np.mean(k_to_accuracies[k]))

    # %% plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

def cross_validation(classfier_module_object, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths, num_iters=1500, batch_size=200):
    """
    use CV to tune the hyperparameters learning-rate for SGD(C) and regularization strength(beta)

    :param classfier_module: this is the instance of a classifer module
    :param X_train: array: n_subject*n_features+1 bias
    :param y_train:
    :param X_val:
    :param y_val:
    :param learning_rates:
    :param regularization_strengths:
    :param num_iters:
    :param batch_size:
    :return:
    """

    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_classfier = None  # The LinearSVM object that achieved the highest validation rate.

    for l in learning_rates:
        for r in regularization_strengths:
            classfier_module_object.train(X_train, y_train, learning_rate=l, reg=r, num_iters=num_iters, batch_size=batch_size)
            y_train_pred = classfier_module_object.predict(X_train)
            y_val_pred = classfier_module_object.predict(X_val)
            training_accuracy = np.mean(y_train == y_train_pred)
            validation_accuracy = np.mean(y_val == y_val_pred)
            results[(l, r)] = (training_accuracy, validation_accuracy)
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_classfier = classfier_module_object

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val

    # %% visualize the CV results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('Validation accuracy')
    plt.show()

    return best_classfier, results

def plot_loss_his(loss_hist, xlab,ylab):
    """
    This is a func to plot the loss history along the different iterations
    :param loss_hist: this is the output after training the training data with a classfier
    :param xlab:
    :param ylab:
    :return:
    """

    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

###############################################################################
################## GRADIENT CHECK
###############################################################################
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print ix, grad[ix]
        it.iternext()  # step to next dimension

    return grad
####################################################################
####### gradient check
####################################################################

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a blob
    into which img_processed_outputs_full_dataset will be written. For example, f might be called like this:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
  sample a few random elements and only return numerical
  in this dimensions.
  Args:
      f: f is a function to get the loss of svm classifier, like a lambda function
      x:
      analytic_grad:
      num_checks:
      h:

  Returns:
    """

    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
##############################################################
################# FEATURE EXTRACTION
##############################################################

def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in xrange(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print 'Done extracting features for %d / %d images' % (i, num_images)

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx / 2::cx, cy / 2::cy].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = plt.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist


pass

#################################################################################################
############################# VISUALIZATION TOOLS
#################################################################################################
def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))  # this is to define the size of the grid to plot the images
    grid_height = H * grid_size + padding * (
    grid_size - 1)  # this is size to store all the 8 * 8 images' value for every pixel, cuz here we define the padding, which will put some blank value between every images<
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))  # this is the images size, C means the GRB channels
    next_idx = 0
    y0, y1 = 0, H
    for y in xrange(grid_size):
        x0, x1 = 0, W
        for x in xrange(grid_size):
            if next_idx < N:
                img = Xs[next_idx]  # here, we have our plotting image
                low, high = np.min(img), np.max(
                    img)  # np.min and max will return the max and min value for the flatten array if axis is not defined
                grid[y0:y1, x0:x1] = ubound * (img - low) / (
                high - low)  # this is to change the tensor value back to image valut[0;255]
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = Xs[n, :, :, :]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  import numpy as np
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def _variable_with_weight_decay(name, shape, stddev, wd=None):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay) ### this is the l2 loss, added into the graphy, later, the cross entropy loss will be added.
  return var


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def onehot(t, num_classes):
    """
    This is a function to convert the label into onehot label
    :param t: numpy array, must be int
    :param num_classes:
    :return:
    """
    # make sure out is int
    t = t.astype(int)
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def save_model(graph_or_sess, fi, out_path):
    """
    Save the given TF session at PATH = "./model/tmp-model"

    :param sess:
        TF sess
    :type sess:  tf.Session object

    :return:
        Path to saved session
    :rtype: String
    """
    if isinstance(graph_or_sess, tf.Graph):
        ops = graph_or_sess.get_operations()
        for op in ops:
            if 'variable' in op.type.lower():
                raise ValueError('Please input a frozen graph (no variables). Or pass in the session object.')

        with graph_or_sess.as_default():
            sess = tf.Session(config=configProto)

            fake_var = tf.Variable([0.0], name="fake_var")
            sess.run(tf.global_variables_initializer())
    else:
        sess=graph_or_sess

    PATH = os.path.join(out_path, "model_" + str(fi), "tmp-model")
    if not os.path.exists(PATH):
	os.makedirs(PATH)
    saver = tf.train.Saver()
    #i should deal with the case in which sess is closed.
    saver.save(sess, PATH)

    if isinstance(graph_or_sess, tf.Graph):
        sess.close()

    return PATH + ".meta"



def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % 'test', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def check_and_clean(d):
  import os, shutil

  if os.path.exists(d):
      shutil.rmtree(d)
  os.mkdir(d)


