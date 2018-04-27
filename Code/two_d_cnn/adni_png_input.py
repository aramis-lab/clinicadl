# coding: utf8

"""
This is the file to handle the ADNI png images, basically, the png file is gray scale image saved as png format and the width is 145, the length is 121.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

author = "Junhao WEN"
copyright = "Copyright 2016-2018 The Aramis Lab Team"
credits = ["Junhao WEN"]
license = "See LICENSE.txt file"
version = "0.1.0"
maintainer = "Junhao WEN"
email = "junhao.wen@inria.fr"
status = "Development"

FLAGS = tf.app.flags.FLAGS

def read_adni_png(filename_queue):
  """Reads and parses examples from ADNI png data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      length: number of rows in the result 121
      width: number of columns in the result 145
      depth: number of color channels in the result 1
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..1.
      uint8image: a [width, length, depth] uint8 Tensor with the image data
  """

  class ADNIPNGRecord(object):
    pass
  result = ADNIPNGRecord()

  # Dimensions of the images in the ADNI png dataset.

  label_bytes = 1
  result.length = FLAGS.png_length
  result.width = FLAGS.png_width
  result.depth = FLAGS.png_depth
  image_bytes = result.length * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the ADNI png format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.width, result.length])
  # Convert from [depth, width, length] to [width, length, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [width, length, 1] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, width, length, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 8
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [FLAGS.batch_size])


def reading_inputs(binary_file):
  """Construct distorted input for ADNI png training using the Reader ops.

  Args:
    FLAGS.data_dir: Path to the ADNI png data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, image_width_rand_crop, image_length_rand_crop, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(FLAGS.data_dir, binary_file)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_adni_png(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  width = FLAGS.image_width_rand_crop
  length = FLAGS.image_length_rand_crop

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # # Randomly crop a [width, length] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [width, length, 1])

  # #Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([width, length, 1])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d ADNI png images before starting to train or test. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, shuffle=True)

