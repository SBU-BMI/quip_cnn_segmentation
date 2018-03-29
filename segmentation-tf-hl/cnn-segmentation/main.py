from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset

import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import arg_scope
import random

from matplotlib import pyplot as plt

import glob
import os
import sys

from config import get_config
#from layers import *

tf.logging.set_verbosity(tf.logging.INFO)

config = None

def normalize(layer):
  return layer/127.5 - 1.

def denormalize(layer):
  return (layer + 1.)/2.

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def seg_model_fn(features, labels, mode, params=config):
  """Create segmentation cnn model"""
  name = None
  #with tf.variable_scope("learner", reuse=False) as sc:
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  #input_layer = tf.reshape(features, [-1, config.input_width, config.input_height, config.input_channel])

  #input_layer = tf.placeholder(
        #tf.float32, [None, self.input_height, self.input_width, self.input_channel])
  #input_layer.set_shape([None, self.input_width, self.input_height, self.input_channel])

  input_layer = tf.reshape(features, [-1, config.input_width, config.input_height, config.input_channel])
  
  layer = slim.conv2d(input_layer, 32, 3, 1, padding='SAME', scope="conv_0")
  layer = slim.conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_1")
  layer = slim.conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_2")
  layr1 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_1")
  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_3")
  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_4")
  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_5")
  layr2 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_2")
  layer = slim.conv2d(layer,128, 3, 1, padding='SAME', scope="conv_6")
  layer = slim.conv2d(layer,128, 3, 1, padding='SAME', scope="conv_7")
  layer = slim.conv2d(layer,128, 3, 1, padding='SAME', scope="conv_8")
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_0");
  layer = tf.concat([layer, layr2], 3)

  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_9")
  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_10")
  layer = slim.conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_11")
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_1");
  layer = tf.concat([layer, layr1], 3)

  layer = slim.conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_12")
  layer = slim.conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_13")
  layer = slim.conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_14")

  logits= slim.conv2d(layer,  1, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_15")
  output = tf.sigmoid(logits, name="sigmoid")

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "masks": tf.sigmoid(logits, name="sigmoid"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
           #logits=logits, labels=tf.to_float(tf.greater(labels,0))), axis=[1, 2], name="loss")
           logits=logits, labels=tf.to_float(tf.greater(labels,0))), name="loss")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    if config.optimizer == "sgd":
      optim = tf.train.GradientDescentOptimizer(config.learner_learning_rate)
    elif config.optimizer == "moment":
      optim = tf.train.MomentumOptimizer(config.learner_learning_rate, 0.95)
    elif config.optimizer == "adam":
      optim = tf.train.AdamOptimizer(config.learner_learning_rate)
    else:
      raise Exception("[!] Unknown optimizer: {}".format(config.optimizer))
    train_op = optim.minimize(
        loss,
        var_list=tf.trainable_variables(),
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["masks"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _read_files_fn(data, mask):
  data_in = cv2.imread(data.decode(), cv2.IMREAD_RGB)
  data_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  mask_in = cv2.imread(mask.decode(), cv2.IMREAD_GRAYSCALE)
  return data_in, mask_in



def main(unused_argv):
  # Load training and eval data
  #train_data, train_labels = get_data(config.train_data_dir, config.train_mask_dir)
#  eval_data, eval_labels = get_data(config.eval_data_dir, config.eval_mask_dir)

  print (config)

  # Create the Estimator
  segmentation_estimator = tf.estimator.Estimator(
      model_fn=seg_model_fn,
      model_dir="model/cnn_segmentation_model",
      params=config
  )

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  #print(type(train_data))
  #print(tf.shape(train_data))
  #print ("data shape ", train_data.shape)
  #print ("data label shape ", train_label.shape)

  def random_crop_image_and_labels(image, label, h, w, is_grayscale):
    combined = tf.concat([image, label], axis=2)
    if is_grayscale:
      combined_crop = tf.random_crop(combined, size=[h, w, 2])
    else:
      combined_crop = tf.random_crop(combined, size=[h, w, 4])
    combined_crop = tf.image.random_flip_left_right(combined_crop)

    num_rot = int(random.random() * 4)
    combined_crop = tf.image.rot90(image=combined_crop, k=num_rot)

    if is_grayscale:
      cropped_image = combined_crop[:, :, :1]
      cropped_label = combined_crop[:, :, 1:]
      cropped_image.set_shape([h, w, 1])
      cropped_label.set_shape([h, w, 1])
    else:
      cropped_image = combined_crop[:, :, :3]
      cropped_label = combined_crop[:, :, 3:]
      cropped_image.set_shape([h, w, 3])
      cropped_label.set_shape([h, w, 1])

    return cropped_image, cropped_label


  def normalize(layer):
    return layer/127.5 - 1.


  def input_fn():
    #print ("Beginning of input_fn")

    image_path_list = list(np.array(glob.glob(os.path.join(config.train_data_dir, '*.png'))))
    image_path_tensor = tf.convert_to_tensor(image_path_list)

    image_paths_ds = Dataset.from_tensor_slices(image_path_tensor)


    image_ds = image_paths_ds.map(
      lambda x: 
    #    tf.image.per_image_standardization( #Normalize data between -1 and 1
        normalize(
          tf.to_float(
              tf.image.decode_png(
                tf.read_file(x)
          )
        )
      )
    )

    # Parallel list of png files in mask dir (should contain files with the same names
    mask_path_list = [config.train_mask_dir + '/' + x.split('/')[-1] 
                        for x in image_path_list]
    mask_path_tensor = tf.convert_to_tensor (mask_path_list)
    mask_paths_ds = Dataset.from_tensor_slices(mask_path_tensor)

    mask_ds = mask_paths_ds.map(lambda x: 
    #  tf.image.per_image_standardization( #Normalize data between -1 and 1
      normalize(
        tf.to_float(
              tf.image.decode_png(
                tf.read_file(x)
          )
        )
      )
    )

    dataset = Dataset.zip((image_ds, mask_ds))
  
    dataset = dataset.map(
      lambda x, y: random_crop_image_and_labels(x, y, config.input_height, config.input_width, config.input_channel == 1)
    )

    dataset = dataset.batch (config.batch_size).shuffle (buffer_size=1000).repeat()

    rv = dataset.make_one_shot_iterator().get_next()

    #print ("Dumping return val from input function")
    #print (rv)

    return rv


  #Use this code to debug input_fn
#  images, labels = input_fn()
#  with tf.Session() as sess:
#    img, label = sess.run ([images, labels])
#    print (img.shape, label.shape)
#    print (img[0])
#    plt.imshow(img[0])
#    plt.show()

#    mask = np.array(label[0]).reshape((140,140))
#    plt.imshow(mask)
#    plt.show()
#  return



  def eval_input_fn():
    image_path_list = list(np.array(glob.glob(os.path.join(config.train_data_dir, '*.png'))))
    image_path_tensor = tf.convert_to_tensor(image_path_list)

    image_paths_ds = Dataset.from_tensor_slices(image_path_tensor)


    image_ds = image_paths_ds.map(
      lambda x: tf.to_float(
            normalize(
              tf.image.decode_png(
                tf.read_file(x)
          )
        )
      )
    )

    # Parallel list of png files in mask dir (should contain files with the same names
    mask_path_list = [config.train_mask_dir + '/' + x.split('/')[-1] 
                        for x in image_path_list]
    mask_path_tensor = tf.convert_to_tensor (mask_path_list)
    mask_paths_ds = Dataset.from_tensor_slices(mask_path_tensor)

    mask_ds = mask_paths_ds.map(lambda x: 
      tf.to_float(
            normalize(
              tf.image.decode_png(
                tf.read_file(x)
          )
        )
      )
    )

    dataset = Dataset.zip((image_ds, mask_ds))
  
    dataset = dataset.map(
      lambda x, y: random_crop_image_and_labels(x, y, config.input_height, config.input_width, config.input_channel == 1)
    )

    dataset = dataset.batch (config.batch_size)

    return dataset.make_one_shot_iterator().get_next()



  segmentation_estimator.train(
      input_fn=input_fn,
      steps=None) # Use None for unlimited steps
      #hooks=[logging_hook])

  # Evaluate the model and print results

#  eval_results = segmentation_estimator.evaluate(input_fn=eval_input_fn)
#  print(eval_results)


if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(argv=[sys.argv[0]] + unparsed)

