from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset

import glob
import os

from config import get_config

tf.logging.set_verbosity(tf.logging.INFO)

config = None

def seg_model_fn(features, labels, mode, config=config):
  #"""Create segmentation cnn model"""
  #with tf.variable_scope("learner", reuse=False) as sc:
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, config.input_width, config.input_height, config.input_channel])
  
  layer = conv2d(input_layer, 32, 3, 1, padding='SAME', scope="conv_0", name=name)
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_1", name=name)
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_2", name=name)
  layr1 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_1")
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_3", name=name)
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_4", name=name)
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_5", name=name)
  layr2 = layer;

  layer = slim.max_pool2d(layer, 3, 2, scope="pool_2")
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_6", name=name)
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_7", name=name)
  layer = conv2d(layer,128, 3, 1, padding='SAME', scope="conv_8", name=name)
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_0");
  layer = tf.concat([layer, layr2], 3)

  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_9", name=name)
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_10", name=name)
  layer = conv2d(layer, 64, 3, 1, padding='SAME', scope="conv_11", name=name)
  layer = slim.conv2d_transpose(layer, 32, 3, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=slim.batch_norm, scope="deconv_1");
  layer = tf.concat([layer, layr1], 3)

  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_12", name=name)
  layer = conv2d(layer, 32, 3, 1, padding='SAME', scope="conv_13", name=name)
  layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_14", name=name)

  logits= conv2d(layer,  1, 3, 1, padding='SAME', normalizer_fn=None, scope="conv_15", name=name)
  output = tf.sigmoid(logits, name="sigmoid")
  #learner_vars = tf.contrib.framework.get_variables(sc)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "masks": tf.sigmoid(logits, name="sigmoid"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  def learner_log_loss(logits, label, name):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.to_float(tf.greater(label,0))), [1, 2], name=name)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = learner_log_loss(logits, label, "learner_loss")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    if config.optimizer == "sgd":
      optim = tf.train.GradientDescentOptimizer(config.learner_learning_rate)
    elif config.optimizer == "moment":
      optim = tf.train.MomentumOptimizer(config.learner_learning_rate, 0.95)
    elif config.optimizer == "adam":
      optim = tf.train.AdamOptimizer(config.learner_learning_rate)
    else:
      raise Exception("[!] Unkown optimizer: {}".format(config.optimizer))
    train_op = optim.minimize(
        loss,
        var_list=learner_vars,
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

  # Create the Estimator
  segmentation_estimator = tf.estimator.Estimator(
      model_fn=seg_model_fn,
      model_dir="model/cnn_segmentation_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  #print(type(train_data))
  #print(tf.shape(train_data))
  #print ("data shape ", train_data.shape)
  #print ("data label shape ", train_label.shape)

  # Train the model
  #train_input_fn = tf.estimator.inputs.numpy_input_fn(
  #    x={"x": train_data},
  #    y=train_labels,
  #    batch_size=100,
  #    num_epochs=None,
  #    shuffle=True)

  def train_input_fn(features, labels, batch_size):

    #print("data length is {}, mask length is {}".format(len(data_path_list), len(mask_path_list)) )

    data_path_list = tf.convert_to_tensor(features)
    mask_path_list = tf.convert_to_tensor(labels)

    dataset = Dataset.from_tensor_slices((data_path_list,mask_path_list))
    dataset = dataset.map(
    lambda data, mask: tuple(tf.py_func(
      _read_files_fn, [data, mask], [tf.float32, tf.float32])))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
    

  # All png files in data dir
  data_path_list = list(np.array(glob.glob(os.path.join(config.train_data_dir, '*.png'))))

  # Parallel list of png files in mask dir (should contain files with the same names
  #mask_path_list = ['/'.join(x.split('/')[:-2]) + '/' + \
  #        config.synthetic_gt_dir + '/' + x.split('/')[-1] for x in data_path_list];
  mask_path_list = [config.train_mask_dir + '/' + x.split('/')[-1] for x in data_path_list]

  segmentation_estimator.train(
      input_fn=lambda:train_input_fn(data_path_list,mask_path_list,config.batch_size),
      steps=20000)
      #hooks=[logging_hook])

#  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
#  eval_results = segmentation_estimator.evaluate(input_fn=eval_input_fn)
#  print(eval_results)


if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run()

