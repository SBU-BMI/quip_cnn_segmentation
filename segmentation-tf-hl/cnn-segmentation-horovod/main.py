from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#from tensorflow.contrib.data import Dataset

import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import arg_scope
import random

from matplotlib import pyplot as plt

import glob
import os
import sys

from config import get_config
#from layers import *

import horovod.tensorflow as hvd

from scipy.misc import imsave

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

def conv2d(inputs, num_outputs, kernel_size, stride,
           padding='SAME',
           layer_dict={}, activation_fn=lrelu,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           normalizer_fn=slim.batch_norm,
           scope=None, **kargv):
  outputs = slim.conv2d(
      inputs, num_outputs, kernel_size,
      stride, activation_fn=activation_fn,
      padding=padding,
      normalizer_fn=normalizer_fn,
      weights_initializer=weights_initializer,
      biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope=scope, **kargv)
  return outputs

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

  input_layer = tf.reshape(features['image'], [-1, config.input_width, config.input_height, config.input_channel])

  layer = conv2d(input_layer, 32, 3, 1, padding='SAME', scope="conv_0")
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_1")
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_2")
  layr1 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_1")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_3")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_4")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_5")
  layr2 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_2")
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_6")
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_7")
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_8")
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_0");
  layer = tf.concat([layer, layr2], 3)

  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_9")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_10")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_11")
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_1");
  layer = tf.concat([layer, layr1], 3)

  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_12")
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_13")
  layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_14")

  logits= conv2d(layer,  1, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_15", activation_fn=None)
  output = tf.sigmoid(logits, name="sigmoid")

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "masks": tf.round(output) * 255,
      # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      #"probabilities": tf.sigmoid(logits, name="sigmoid_tensor"),
      # Can't forward features this way, need to use forward_features() as wrapper to estimator object instead
      #"in_file": features['in_file']
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
           #logits=logits, labels=tf.to_float(tf.greater(labels,0))), axis=[1, 2], name="loss")
           logits=logits, labels=tf.to_float(tf.greater(labels,0))), name="loss")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Horovod: scale learning rate by the number of workers.
    rate = config.learner_learning_rate * hvd.size()
    if config.optimizer == "sgd":
      optim = tf.train.GradientDescentOptimizer(rate)
    elif config.optimizer == "moment":
      optim = tf.train.MomentumOptimizer(rate, 0.95)
    elif config.optimizer == "adam":
      optim = tf.train.AdamOptimizer(rate)
    else:
      raise Exception("[!] Unknown optimizer: {}".format(config.optimizer))

    # Horovod: add Horovod Distributed Optimizer.
    optim = hvd.DistributedOptimizer(optim)

    train_op = optim.minimize(
        loss,
        var_list=tf.trainable_variables(),
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.to_float(tf.greater(labels,0)),
          predictions=tf.to_float(tf.greater(predictions["masks"],0.5)))}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _read_files_fn(data, mask):
  data_in = cv2.imread(data.decode(), cv2.IMREAD_RGB)
  data_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  mask_in = cv2.imread(mask.decode(), cv2.IMREAD_GRAYSCALE)
  return data_in, mask_in



def main(unused_argv):

  #print (config)

  hvd.init()

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  cfg = tf.ConfigProto()
  cfg.gpu_options.allow_growth = True
  cfg.gpu_options.visible_device_list = str(hvd.local_rank())

  model_dir="model/cnn_segmentation_model" if hvd.rank() == 0 else None

  # Create the Estimator
  segmentation_estimator = tf.estimator.Estimator(
      model_fn=seg_model_fn,
      model_dir=model_dir,
      params=config,
      config=tf.estimator.RunConfig(session_config=cfg)
  )

  #Include the file names of input images in the prediction results so we can write out the mask files later
  segmentation_estimator = tf.contrib.estimator.forward_features (segmentation_estimator, 'in_file')

  # Set up logging for predictions
  # Log the values in the "sigmoid" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "sigmoid_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)


  def random_crop_image_and_labels(features, label, h, w, is_grayscale):
    combined = tf.concat([features['image'], label], axis=2)
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

    return {'image': cropped_image, 'in_file': features['in_file']}, cropped_label


  # Input function for use with segmentation_estimator.train()
  def input_fn():
    image_path_list = list(np.array(glob.glob(os.path.join(config.train_data_dir, '*.png'))))
    image_path_tensor = tf.convert_to_tensor(image_path_list, dtype=tf.string)
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_path_tensor)
    image_ds = image_paths_ds.map(
      lambda name: { 'image': normalize(
          tf.to_float(
              tf.image.decode_png(
                tf.read_file(name)
          )
        )
      ), 'in_file': name})
    # Parallel list of png files in mask dir (should contain files with the same names
    mask_path_list = [config.train_mask_dir + '/' + x.split('/')[-1]
                        for x in image_path_list]
    mask_path_tensor = tf.convert_to_tensor (mask_path_list, dtype=tf.string)
    mask_paths_ds = tf.data.Dataset.from_tensor_slices(mask_path_tensor)
    mask_ds = mask_paths_ds.map(lambda x:
    #  tf.image.per_image_standardization( #Normalize data between -1 and 1
        tf.to_float(
              tf.image.decode_png(
                tf.read_file(x)
          )
      )
    )
    dataset = tf.data.Dataset.zip((image_ds, mask_ds))
    dataset = dataset.map(
      lambda x, y: random_crop_image_and_labels(x, y, config.input_height, config.input_width, config.input_channel == 1)
    )
    dataset = dataset.batch (config.batch_size).shuffle (buffer_size=1000).repeat()
    rv = dataset.make_one_shot_iterator().get_next()
    return rv



  # Input function for use with segmentation_estimator.evaluate()
  def eval_input_fn():
    image_path_list = list(np.array(glob.glob(os.path.join(config.train_data_dir, '*.png'))))
    image_path_tensor = tf.convert_to_tensor(image_path_list, dtype=tf.string)
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_path_tensor)
    image_ds = image_paths_ds.map(
      lambda name: { 'image': normalize(
            tf.to_float(
              tf.image.decode_png(
                tf.read_file(name)
          )
        )
      ), 'in_file': name
      })
    # Parallel list of png files in mask dir (should contain files with the same names
    mask_path_list = [config.train_mask_dir + '/' + x.split('/')[-1]
                        for x in image_path_list]
    mask_path_tensor = tf.convert_to_tensor (mask_path_list, dtype=tf.string)
    mask_paths_ds = tf.data.Dataset.from_tensor_slices(mask_path_tensor)
    mask_ds = mask_paths_ds.map(lambda x:
      tf.to_float(
              tf.image.decode_png(
                tf.read_file(x)
        )
      )
    )
    dataset = tf.data.Dataset.zip((image_ds, mask_ds))
    dataset = dataset.map(
      lambda x, y: random_crop_image_and_labels(x, y, config.input_height, config.input_width, config.input_channel == 1)
    )
    dataset = dataset.batch (config.batch_size)
    return dataset.make_one_shot_iterator().get_next()


  # Input function for use with segmentation_estimator.predict()
  def predict_input_fn():
    image_path_list = list(np.array(glob.glob(os.path.join(config.real_image_dir, '*.png'))))
    image_path_tensor = tf.convert_to_tensor(image_path_list, dtype=tf.string)
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_path_tensor)
    image_ds = image_paths_ds.map(
      lambda name: {'image': normalize(
            tf.to_float(
              tf.image.decode_png(
                tf.read_file(name)
          )
        )
      ), 'in_file': [name]}
    )
    # This is broken since it ignores the image/in_file dictionary. Just commenting out as images should already be the right size
    #image_ds = image_ds.map(
    #  lambda img: tf.image.crop_to_bounding_box(img, 0, 0, config.input_height, config.input_width)
    #)
    return image_ds.make_one_shot_iterator().get_next()


  #Use this code to debug input_fn
#  features = predict_input_fn()
#  with tf.Session() as sess:
#    feat = sess.run ([features])
#    print (feat[0])
#    print ("::::::", feat[0]['in_file'])
#    print (feat[0]['image'].shape)
#    plt.imshow(feat[0]['image'][0])
#    plt.show()

#  return



  # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
  # rank 0 to all other processes. This is necessary to ensure consistent
  # initialization of all workers when training is started with random weights or
  # restored from a checkpoint.
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

  if config.is_train:
    segmentation_estimator.train(
      input_fn=input_fn,
      steps=500, # Use None for unlimited steps
      hooks=[logging_hook, bcast_hook])
    # Evaluate the model and print results
    eval_results = segmentation_estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

  else:
    predictions = segmentation_estimator.predict(
      input_fn=predict_input_fn
    )
    pl = list(predictions)
    for pred in pl:
      #print ("Writing:")
      #print (pred['in_file'])
      #print (pred['masks'].shape)
      imsave(pred['in_file'][:-4]+"_mask.png", np.array(pred['masks'][:,:,0]), format="png")


if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(argv=[sys.argv[0]] + unparsed)

