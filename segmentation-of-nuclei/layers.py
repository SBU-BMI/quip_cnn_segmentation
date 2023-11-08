import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import add_arg_scope
from utils import synthetic_to_refer_paths, synthetic_to_ground_truth_paths, supervised_to_ground_truth_paths
import glob

SE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits

def int_shape(x):
  return list(map(int, x.get_shape()[1: ]))

def normalize(layer):
  return layer/127.5 - 1.

def denormalize(layer):
  return (layer + 1.)/2.

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def identity(x):
    return x;

def _update_dict(layer_dict, scope, layer):
  name = "{}/{}".format(tf.get_variable_scope().name, scope)
  layer_dict[name] = layer

def image_from_paths(fake_image_path, config, is_grayscale=False):
  fake_image_path_list = list(fake_image_path);
  fake_ground_truth_list = synthetic_to_ground_truth_paths(fake_image_path_list, config);
  refer_image_list = synthetic_to_refer_paths(fake_image_path_list, config);

  filename_queue, mask_queue, filename_refer_queue = \
          tf.train.slice_input_producer([tf.convert_to_tensor(fake_image_path_list),
                    tf.convert_to_tensor(fake_ground_truth_list), tf.convert_to_tensor(refer_image_list)])

  image_data = tf.read_file(filename_queue)
  mask_data = tf.read_file(mask_queue)
  refer_data = tf.read_file(filename_refer_queue)

  image = tf.image.decode_png(image_data, channels=3, dtype=tf.uint8)
  mask = tf.image.decode_png(mask_data, channels=1, dtype=tf.uint8)
  refer = tf.image.decode_png(refer_data, channels=3, dtype=tf.uint8)

  if is_grayscale:
    image = tf.image.rgb_to_grayscale(image)
    refer = tf.image.rgb_to_grayscale(refer)

  return filename_queue, tf.to_float(image), tf.to_float(refer), tf.to_float(mask)

def supervised_from_paths(supervised_image_path, config, is_grayscale=False):
  supervised_image_path_list = list(supervised_image_path);
  supervised_ground_truth_list = supervised_to_ground_truth_paths(supervised_image_path_list, config);

  filename_queue, mask_queue = \
          tf.train.slice_input_producer([tf.convert_to_tensor(supervised_image_path_list),
                    tf.convert_to_tensor(supervised_ground_truth_list)])

  image_data = tf.read_file(filename_queue)
  mask_data = tf.read_file(mask_queue)

  image = tf.image.decode_png(image_data, channels=3, dtype=tf.uint8)
  mask = tf.image.decode_png(mask_data, channels=1, dtype=tf.uint8)

  if is_grayscale:
    image = tf.image.rgb_to_grayscale(image)

  return tf.to_float(image), tf.to_float(mask)

@add_arg_scope
def resnet_block(
    inputs, scope, num_outputs=64, kernel_size=[5, 5], normalizer_fn=None,
    stride=[1, 1], padding="SAME", layer_dict={}):
  with tf.variable_scope(scope):
    layer = conv2d(
        inputs, num_outputs, kernel_size, stride, normalizer_fn=normalizer_fn,
        padding=padding, activation_fn=lrelu, scope="conv1")
    layer = conv2d(
        layer, num_outputs, kernel_size, stride, normalizer_fn=normalizer_fn,
        padding=padding, activation_fn=lrelu, scope="conv2")
    if inputs.get_shape()[1] == layer.get_shape()[1] and \
       inputs.get_shape()[2] == layer.get_shape()[2] and \
       inputs.get_shape()[3] == layer.get_shape()[3]:
      outputs = tf.add(inputs, layer)
    else:
      print("Cannot tf.add: shapes do not match {}-{} {}-{} {}-{}".format(
              inputs.get_shape()[1], layer.get_shape()[1],
              inputs.get_shape()[2], layer.get_shape()[2],
              inputs.get_shape()[3], layer.get_shape()[3]));
      outputs = layer;

  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def repeat(inputs, repetitions, layer, layer_dict={}, **kargv):
  outputs = slim.repeat(inputs, repetitions, layer, **kargv)
  _update_dict(layer_dict, kargv['scope'], outputs)
  return outputs

@add_arg_scope
def conv2d(inputs, num_outputs, kernel_size, stride,
           padding='SAME',
           layer_dict={}, activation_fn=lrelu,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           normalizer_fn=slim.batch_norm,
           scope=None, name="", **kargv):
  outputs = slim.conv2d(
      inputs, num_outputs, kernel_size,
      stride, activation_fn=activation_fn,
      padding=padding,
      normalizer_fn=normalizer_fn,
      weights_initializer=weights_initializer,
      biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope=scope, **kargv)
  if name:
    scope = "{}/{}".format(name, scope)
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def dense(inputs, num_outputs,
           layer_dict={}, activation_fn=lrelu,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           scope=None, name="", **kargv):
  outputs = slim.fully_connected(
      inputs, num_outputs,
      activation_fn=activation_fn,
      weights_initializer=weights_initializer, scope=scope, **kargv)
  if name:
    scope = "{}/{}".format(name, scope)
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def max_pool2d(inputs, kernel_size=[3, 3], stride=[1, 1],
               layer_dict={}, scope=None, name="", **kargv):
  outputs = slim.max_pool2d(inputs, kernel_size, stride, **kargv)
  if name:
    scope = "{}/{}".format(name, scope)
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def tanh(inputs, layer_dict={}, name=None, **kargv):
  outputs = tf.nn.tanh(inputs, name=name, **kargv)
  _update_dict(layer_dict, name, outputs)
  return outputs
