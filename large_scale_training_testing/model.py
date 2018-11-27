import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import arg_scope
import random

from layers import *
from utils import show_all_variables

class Model(object):
  def __init__(self, config, data_loader):
    self.data_loader = data_loader

    self.debug = config.debug
    self.config = config

    self.input_height = config.input_height
    self.input_width = config.input_width
    self.input_channel = config.input_channel

    self.real_scale = config.real_scale
    self.reg_scale_l1 = config.reg_scale_l1
    self.reg_scale_l2 = config.reg_scale_l2
    self.learner_adv_scale = config.learner_adv_scale
    self.refiner_learning_rate = config.refiner_learning_rate
    self.discrim_learning_rate = config.discrim_learning_rate
    self.learner_learning_rate = config.learner_learning_rate
    self.max_grad_norm = config.max_grad_norm

    if config.with_batch_norm:
        self.bn = slim.batch_norm;
    else:
        self.bn = None;

    self.layer_dict = {}

    roi_weights = np.zeros((1, self.input_height, self.input_width, 1), dtype=np.float32) + 0.01
    roi_weights[:, 50:-50, 50:-50 :] = 1.0
    self.learner_roi_constant = tf.convert_to_tensor(roi_weights, dtype=tf.float32)
    self._build_placeholders()
    self._build_model()
    self._build_steps()
    self._build_optim()

    show_all_variables()

  def random_crop_image_and_labels(self, image, label, h, w, is_grayscale):
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

  def random_crop(self, image, h, w, is_grayscale):
    if is_grayscale:
      cropped = tf.random_crop(image, size=[h, w, 1])
    else:
      cropped = tf.random_crop(image, size=[h, w, 3])
    cropped = tf.image.random_flip_left_right(cropped)

    num_rot = int(random.random() * 4)
    cropped = tf.image.rot90(image=cropped, k=num_rot)
    if is_grayscale:
      cropped.set_shape([h, w, 1])
    else:
      cropped.set_shape([h, w, 3])

    return cropped;

  def deter_crop(self, image, h, w, is_grayscale):
    cropped_image=image[0:h, 0:w, :]
    if is_grayscale:
        cropped_image.set_shape([h, w, 1])
    else:
        cropped_image.set_shape([h, w, 3])
    return cropped_image

  def _build_placeholders(self):
    image_dims = [self.input_height, self.input_width, self.input_channel]
    is_grayscale = (self.input_channel==1);

    min_after_dequeue = 5000

    self.synthetic_batch_size = tf.placeholder(tf.int32, [], "synthetic_batch_size")
    self.synthetic_filenames, self.synthetic_images, self.ref_image, self.mask = \
        image_from_paths(self.data_loader.synthetic_data_paths, self.config, is_grayscale=is_grayscale);
    self.supervised_image, self.supervised_mask = \
        supervised_from_paths(self.data_loader.supervised_image_path, self.config, is_grayscale=is_grayscale);

    self.synthetic_images, self.mask = self.random_crop_image_and_labels(
          self.synthetic_images, self.mask, self.input_height, self.input_width, is_grayscale);
    self.ref_image = self.random_crop(self.ref_image, self.input_height, self.input_width, is_grayscale)

    self.supervised_image, self.supervised_mask = self.random_crop_image_and_labels(
          self.supervised_image, self.supervised_mask, self.input_height, self.input_width, is_grayscale);

    self.x_filename, self.x, self.xmask, self.ref_image = tf.train.shuffle_batch(
          [self.synthetic_filenames, self.synthetic_images, self.mask, self.ref_image],
          batch_size=self.synthetic_batch_size,
          num_threads=8, capacity=min_after_dequeue+3*self.config.batch_size,
          min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    self.rx, self.rxmask = tf.train.shuffle_batch(
          [self.supervised_image, self.supervised_mask],
          batch_size=self.config.sup_batch_size,
          num_threads=8, capacity=min_after_dequeue+3*self.config.sup_batch_size,
          min_after_dequeue=min_after_dequeue, name='supervised_inputs')

    # Normalized test_patch directly cropped from testing patches
    self.test_patch_normalized = tf.placeholder(
        tf.float32, [None, None, None, self.input_channel])

    # Load real y from data_loader
    self.y = tf.placeholder(
        tf.uint8, [None, self.input_height, self.input_width, self.input_channel], name='real_inputs')
    self.normalized_y = normalize(tf.cast(self.y, tf.float32))

    # Load real reference y from data_loader
    self.ref_y = tf.placeholder(
        tf.uint8, [None, self.input_height, self.input_width, self.input_channel], name='ref_real_inputs')
    self.normalized_ref_y = normalize(tf.cast(self.ref_y, tf.float32))

    # Load history of refiner output from buffer
    self.R_x_history = tf.placeholder(
        tf.float32, [None, self.input_height, self.input_width, self.input_channel], 'R_x_history')

    # Load history of mask from buffer
    self.mask_history = tf.placeholder(
        tf.float32, [None, self.input_height, self.input_width, 1], 'mask_history')

    # Load history of reference image from buffer
    self.refimg_history = tf.placeholder(
        tf.float32, [None, self.input_height, self.input_width, self.input_channel],
        'refimg_history')
    self.normalized_refimg_history = normalize(self.refimg_history)

    # Load x from images on hard drive
    self.normalized_x = normalize(self.x)

    # Load rx from images on hard drive
    self.normalized_rx = normalize(self.rx)

    # Load reference images (in fake pairs) from images on hard drive
    self.normalized_ref_image = normalize(self.ref_image)

    self.refiner_step = tf.Variable(0, name='refiner_step', trainable=False)
    self.discrim_step = tf.Variable(0, name='discrim_step', trainable=False)
    self.learner_step = tf.Variable(0, name='learner_step', trainable=False)

  def _build_optim(self):
    def minimize(loss, lrate, step, var_list):
      if self.config.optimizer == "sgd":
        optim = tf.train.GradientDescentOptimizer(lrate)
      elif self.config.optimizer == "moment":
        optim = tf.train.MomentumOptimizer(lrate, 0.95)
      elif self.config.optimizer == "adam":
        optim = tf.train.AdamOptimizer(lrate)
      else:
        raise Exception("[!] Unkown optimizer: {}".format(self.config.optimizer))

      if self.max_grad_norm != None:
        grads_and_vars = optim.compute_gradients(loss)
        new_grads_and_vars = []
        for idx, (grad, var) in enumerate(grads_and_vars):
          if grad is not None and var in var_list:
            new_grads_and_vars.append((tf.clip_by_norm(grad, self.max_grad_norm), var))
        return optim.apply_gradients(new_grads_and_vars, global_step=step)
      else:
        return optim.minimize(loss, global_step=step, var_list=var_list)

    self.refiner_optim = minimize(
        self.refiner_loss, self.refiner_learning_rate, self.refiner_step, self.refiner_vars)

    self.discrim_optim = minimize(
        self.discrim_loss, self.discrim_learning_rate, self.discrim_step, self.discrim_vars)

    self.discrim_optim_with_history = minimize(
        self.discrim_loss_with_history, self.discrim_learning_rate, self.discrim_step, self.discrim_vars)

    self.learner_optim = minimize(
          self.learner_loss, self.learner_learning_rate, self.learner_step, self.learner_vars)

  def _build_model(self):
    with arg_scope([resnet_block, conv2d, max_pool2d, dense, tanh,],
                   layer_dict=self.layer_dict):
      self.R_x = self._build_refiner(self.normalized_x, self.normalized_ref_image)
      self.denormalized_R_x = denormalize(self.R_x)

      self.L_R_x, self.L_R_x_logits = \
              self._build_learner(self.R_x, name="L_R_x")
      self.L_rx, self.L_rx_logits = \
              self._build_learner(self.normalized_rx, name="L_rx", reuse=True)
      self.L_test_patch, self.L_patch_logits = \
              self._build_learner(self.test_patch_normalized, name="L_test_patch", reuse=True)
      self.L_R_x_history, self.L_R_x_history_logits = \
              self._build_learner(self.R_x_history, name="L_R_x_history", reuse=True)

      self.D_y, self.D_y_logits = \
          self._build_discrim(self.normalized_y, self.normalized_ref_y, name="D_y")
      self.D_R_x, self.D_R_x_logits = \
          self._build_discrim(self.R_x, self.normalized_ref_image, name="D_R_x", reuse=True)
      self.D_R_x_history, self.D_R_x_history_logits = \
          self._build_discrim(self.R_x_history, self.normalized_refimg_history, name="D_R_x_history", reuse=True)

    self._build_loss()

  def _build_loss(self):
    # Refiner loss
    def fake_label(layer):
      return tf.zeros_like(layer, dtype=tf.int32)[:,:,:,0]

    def real_label(layer):
      return tf.ones_like(layer, dtype=tf.int32)[:,:,:,0]

    def log_loss(logits, label, name):
      return tf.reduce_sum(SE_loss(logits=logits, labels=label), [1, 2], name=name)

    def learner_log_loss(logits, label, importance, name):
        return learner_seg_loss(logits, label, importance, name+'_seg') \
                + learner_det_loss(logits, label, importance, name+'_det')

    def learner_seg_loss(logits, label, importance, name):
      bin_label = tf.to_float(tf.greater(tf.floormod(label+0.01, 2), 0.9))
      if self.config.importance_sampling and (importance is not None):
        importance = tf.reduce_mean(importance, [1, 2])[:, 1]
        importance = importance / (1 - importance + 0.000001)
        importance = tf.clip_by_value(importance,
                clip_value_min=self.config.importance_minimum,
                clip_value_max=self.config.importance_maximum)
        xentro = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[..., 1:], labels=bin_label) * self.learner_roi_constant,
            [1, 2, 3], name=name)
        return xentro * importance
      else:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[..., 1:], labels=bin_label) * self.learner_roi_constant,
            [1, 2, 3], name=name)

    def learner_det_loss(logits, label, importance, name):
      bin_label = tf.nn.dilation2d(tf.to_float(tf.greater(label, 3.5)),
              filter=tf.ones([3,3,1]), strides=(1,1,1,1), rates=[1,1,1,1], padding='SAME', name='dilated_gt')
      if self.config.importance_sampling and (importance is not None):
        importance = tf.reduce_mean(importance, [1, 2])[:, 1]
        importance = importance / (1 - importance + 0.000001)
        importance = tf.clip_by_value(importance,
                clip_value_min=self.config.importance_minimum,
                clip_value_max=self.config.importance_maximum)
        xentro = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[..., :1], labels=tf.to_float(tf.greater(bin_label, 1.5))) * self.learner_roi_constant,
            [1, 2, 3], name=name)
        return xentro * importance
      else:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[..., :1], labels=tf.to_float(tf.greater(bin_label, 1.5))) * self.learner_roi_constant,
            [1, 2, 3], name=name)

    # Error tracking
    self.refiner_err = tf.reduce_sum(
            tf.cast(tf.is_nan(self.R_x), tf.float32) * \
            tf.cast(tf.is_inf(self.R_x), tf.float32), name="refiner_err")
    self.discrim_err = tf.reduce_sum(
            tf.cast(tf.is_nan(self.D_R_x_logits), tf.float32) * \
            tf.cast(tf.is_inf(self.D_R_x_logits), tf.float32), name="discrim_err")

    with tf.name_scope("learner"):
      self.learner_loss_hist = learner_log_loss(self.L_R_x_history_logits, self.mask_history, self.D_R_x_history, "learner_loss_history")
      self.learner_loss_ind = learner_log_loss(self.L_R_x_logits, self.xmask, self.D_R_x, "learner_loss_ind")
      self.learner_sup_loss = learner_log_loss(self.L_rx_logits, self.rxmask, None, "learner_sup_loss")

      self.learner_loss = tf.reduce_mean(
              tf.concat([self.learner_loss_hist, self.learner_loss_ind, self.learner_sup_loss], axis=0), name="learner_loss");

      self.learner_summary = tf.summary.merge([
          tf.summary.scalar("learner/learner_loss", tf.reduce_mean(self.learner_loss)),
          tf.summary.scalar("learner/refiner_err", tf.reduce_mean(self.refiner_err)),
          tf.summary.scalar("learner/discrim_err", tf.reduce_mean(self.discrim_err)),
      ])

    with tf.name_scope("refiner"):
      self.realism_loss = self.real_scale * log_loss(
          self.D_R_x_logits, real_label(self.D_R_x_logits), "realism_loss")
      self.reg_loss_l1 = self.reg_scale_l1 * tf.reduce_sum(
              tf.abs(self.R_x - self.normalized_x), [1, 2, 3], name="regularization_loss_l1");
      self.reg_loss_l2 = self.reg_scale_l2 * tf.reduce_sum(
              tf.square(self.R_x - self.normalized_x), [1, 2, 3], name="regularization_loss_l2");
      self.regularization_loss = self.reg_loss_l1 + self.reg_loss_l2;
      self.learner_loss_adv = learner_log_loss(self.L_R_x_logits, self.xmask, importance=None, name="learner_loss_adv")

      self.refiner_loss = tf.reduce_mean(
              self.realism_loss + self.regularization_loss - self.learner_adv_scale * self.learner_loss_adv,
              name="refiner_loss");

      if self.debug:
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.R_x], "R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.D_R_x], "D_R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.normalized_x], "normalized_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.denormalized_R_x], "denormalized_R_x")
        self.refiner_loss = tf.Print(
            self.refiner_loss, [self.regularization_loss], "reg_loss")

      self.refiner_summary = tf.summary.merge([
          tf.summary.scalar("refiner/realism_loss", tf.reduce_mean(self.realism_loss)),
          tf.summary.scalar("refiner/reg_loss_l1", tf.reduce_mean(self.reg_loss_l1)),
          tf.summary.scalar("refiner/reg_loss_l2", tf.reduce_mean(self.reg_loss_l2)),
          tf.summary.scalar("refiner/loss", tf.reduce_mean(self.refiner_loss)),
          tf.summary.scalar("refiner/refiner_err", tf.reduce_mean(self.refiner_err)),
          tf.summary.scalar("refiner/discrim_err", tf.reduce_mean(self.discrim_err)),
      ])

    # Discriminator loss
    with tf.name_scope("discriminator"):
      # Loss without history
      self.refiner_d_loss = log_loss(
          self.D_R_x_logits, fake_label(self.D_R_x_logits), "refiner_d_loss")
      self.synthetic_d_loss = log_loss(
          self.D_y_logits, real_label(self.D_y_logits), "synthetic_d_loss")
      self.discrim_loss = tf.reduce_mean(self.refiner_d_loss + self.synthetic_d_loss, name="discrim_loss")

      # Compute accuracy for tensorboard tracking
      self.refiner_d_acc = tf.metrics.accuracy(
          tf.argmax(self.D_R_x_logits, axis=3), fake_label(self.D_R_x_logits), name="refiner_d_acc")
      self.synthetic_d_acc = tf.metrics.accuracy(
          tf.argmax(self.D_y_logits, axis=3), real_label(self.D_y_logits), name="synthetic_d_acc")

      # Loss with history
      self.refiner_d_loss_with_history = log_loss(
          self.D_R_x_history_logits, fake_label(self.D_R_x_history_logits), "refiner_d_loss_with_history")
      self.discrim_loss_with_history = tf.reduce_mean(
          tf.concat([self.refiner_d_loss, self.refiner_d_loss_with_history], axis=0) + \
              self.synthetic_d_loss, name="discrim_loss_with_history")

      if self.debug:
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_R_x_logits], "D_R_x_logits")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_y_logits], "D_y_logits")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.refiner_d_loss], "refiner_d_loss")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.refiner_d_loss_with_history], "refiner_d_loss_with_history")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.synthetic_d_loss], "synthetic_d_loss")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_R_x_history_logits], "D_R_x_history_logits")
        self.discrim_loss_with_history = tf.Print(
            self.discrim_loss_with_history, [self.D_y_logits], "D_y_logits")

      self.discrim_summary = tf.summary.merge([
          tf.summary.scalar("discrim/synthetic_d_loss", tf.reduce_mean(self.synthetic_d_loss)),
          tf.summary.scalar("discrim/refiner_d_loss", tf.reduce_mean(self.refiner_d_loss)),
          tf.summary.scalar("discrim/discrim_loss", tf.reduce_mean(self.discrim_loss)),
          tf.summary.scalar("discrim/refiner_d_acc", tf.reduce_mean(self.refiner_d_acc)),
          tf.summary.scalar("discrim/synthetic_d_acc", tf.reduce_mean(self.synthetic_d_acc)),
          tf.summary.scalar("discrim/refiner_err", tf.reduce_mean(self.refiner_err)),
          tf.summary.scalar("discrim/discrim_err", tf.reduce_mean(self.discrim_err)),
      ])
      self.discrim_summary_with_history = tf.summary.merge([
          tf.summary.scalar("discrim/synthetic_d_loss", tf.reduce_mean(self.synthetic_d_loss)),
          tf.summary.scalar("discrim/refiner_d_loss_with_history", tf.reduce_mean(self.refiner_d_loss_with_history)),
          tf.summary.scalar("discrim/discrim_loss_with_history", tf.reduce_mean(self.discrim_loss_with_history)),
          tf.summary.scalar("discrim/refiner_d_acc", tf.reduce_mean(self.refiner_d_acc)),
          tf.summary.scalar("discrim/synthetic_d_acc", tf.reduce_mean(self.synthetic_d_acc)),
          tf.summary.scalar("discrim/refiner_err", tf.reduce_mean(self.refiner_err)),
          tf.summary.scalar("discrim/discrim_err", tf.reduce_mean(self.discrim_err)),
      ])

  def _build_steps(self):
    def run(sess, feed_dict, fetch,
            summary_op, summary_writer, output_op=None):
      if summary_writer is not None:
        fetch['summary'] = summary_op
      if output_op is not None:
        fetch['output'] = output_op

      result = sess.run(fetch, feed_dict=feed_dict)
      if result.has_key('summary'):
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train_refiner(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.refiner_loss,
          'optim': self.refiner_optim,
          'step': self.refiner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.refiner_summary, summary_writer,
                 output_op=[self.R_x, self.xmask, self.ref_image] if with_output else None)

    def test_refiner(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          'filename': self.x_filename,
          'loss': self.refiner_loss,
          'step': self.refiner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.refiner_summary, summary_writer,
                 output_op=[self.x, self.R_x, self.xmask, self.x_filename] if with_output else None)

    def train_discrim(sess, feed_dict, summary_writer=None,
                      with_history=False, with_output=False):
      fetch = {
          'loss': self.discrim_loss_with_history,
          'optim': self.discrim_optim_with_history,
          'step': self.discrim_step,
      }
      return run(sess, feed_dict, fetch,
                 self.discrim_summary_with_history if with_history \
                     else self.discrim_summary, summary_writer,
                 output_op=self.D_R_x if with_output else None)

    def train_learner(sess, feed_dict, summary_writer=None,
                      with_history=False, with_output=False):
      fetch = {
          'loss': self.learner_loss,
          'optim': self.learner_optim,
          'step': self.learner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.learner_summary ,summary_writer,
                 output_op=self.L_R_x_history if with_output else None)


    def test_discrim(sess, feed_dict, summary_writer=None,
                     with_history=False, with_output=False):
      fetch = {
          'loss': self.discrim_loss,
          'step': self.discrim_step,
      }
      return run(sess, feed_dict, fetch,
                 self.discrim_summary_with_history if with_history \
                     else self.discrim_summary, summary_writer,
                 output_op=self.D_R_x if with_output else None)

    def test_learner_patch(sess, feed_dict, fetch, summary_writer=None,
                           with_output=False):
      fetch = {
          'step': self.learner_step,
      }
      return run(sess, feed_dict, fetch,
                 self.learner_summary, summary_writer,
                 output_op=self.L_test_patch if with_output else None)

    self.train_refiner = train_refiner
    self.test_refiner = test_refiner
    self.train_discrim = train_discrim
    self.test_discrim = test_discrim
    self.train_learner = train_learner
    self.test_learner_patch = test_learner_patch

  def _build_refiner(self, synthesized, rlyer):
    with tf.variable_scope("refiner") as sc:
      layer = conv2d(synthesized, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="pre_conv_0");

      if self.config.with_ref:
        rlyer = conv2d(rlyer, 32, 4, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_0");
        rlyer = slim.avg_pool2d(rlyer, 3, 2, scope="ref_pool_0")
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_1");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_2");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_3");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_4");
        pool_height = (self.input_height-3-1)/2-4-4-4-4;
        pool_width = (self.input_width-3-1)/2-4-4-4-4;
        rlyer = slim.avg_pool2d(rlyer, (pool_height, pool_width), scope="ref_pool_1");
        rlyer = tf.tile(rlyer, [1, self.input_height, self.input_width, 1]);
        layer = tf.concat([layer, rlyer], 3)

      layer = repeat(layer, 5, resnet_block, num_outputs=32, padding='SAME', normalizer_fn=self.bn, scope="resnet")
      layer = conv2d(layer, 3, 3, 1, padding='SAME', normalizer_fn=None, activation_fn=identity, scope="conv_1");

      output = tf.clip_by_value(synthesized+layer,
               clip_value_min=-1.0, clip_value_max=1.0, name="refiner_output_clip")
      self.refiner_vars = tf.contrib.framework.get_variables(sc)
    return output

  def _build_discrim(self, layer, rlyer, name, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse) as sc:
      layer = conv2d(layer, 32, 3, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_0", name=name)

      if self.config.with_ref:
        rlyer = conv2d(rlyer, 32, 4, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_0");
        rlyer = slim.avg_pool2d(rlyer, 3, 2, scope="ref_pool_0")
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_1");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_2");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_3");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="ref_conv_4");
        pool_height = (self.input_height-3-1)/2-4-4-4-4;
        pool_width = (self.input_width-3-1)/2-4-4-4-4;
        rlyer = slim.avg_pool2d(rlyer, (pool_height, pool_width), scope="ref_pool_1");
        rlyer = tf.tile(rlyer, [1, self.input_height-2, self.input_width-2, 1]);
        layer = tf.concat([layer, rlyer], 3)

      layer = conv2d(layer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_0_0", name=name)
      layer = conv2d(layer, 32, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_0_1", name=name)
      layer = slim.avg_pool2d(layer, 2, 2, scope="pool_0")

      layer = conv2d(layer, 48, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_1_0", name=name)
      layer = conv2d(layer, 48, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_1_1", name=name)
      layer = conv2d(layer, 48, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_1_2", name=name)
      layer = slim.avg_pool2d(layer, 3, 2, scope="pool_1")

      layer = conv2d(layer, 64, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_2_0", name=name)
      layer = conv2d(layer, 64, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_2_1", name=name)
      layer = conv2d(layer, 64, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_2_2", name=name)
      layer = slim.avg_pool2d(layer, 3, 2, scope="pool_2")

      layer = conv2d(layer, 80, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_3_0", name=name)
      layer = conv2d(layer, 80, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_3_1", name=name)
      layer = conv2d(layer, 80, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_3_2", name=name)
      layer = slim.avg_pool2d(layer, 3, 2, scope="pool_3")

      layer = conv2d(layer,128, 5, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_4_0", name=name)
      layer = conv2d(layer,128, 3, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_4_1", name=name)
      layer = conv2d(layer,128, 3, 1, padding='VALID', normalizer_fn=self.bn, scope="conv_4_2", name=name)
      logits = conv2d(layer, 2, 1, 1, padding='VALID', normalizer_fn=None, activation_fn=identity, scope="conv_5", name=name)
      output = tf.nn.softmax(logits, name="softmax")
      self.discrim_vars = tf.contrib.framework.get_variables(sc)
    return output, logits

  def _build_learner(self, layer, name, reuse=False):
    with tf.variable_scope("learner", reuse=reuse) as sc:
      layer.set_shape([None, self.input_width, self.input_height, self.input_channel])

      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_0_0", name=name)
      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_0_1", name=name)
      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_0_2", name=name)
      layr1 = layer;

      layer = slim.max_pool2d(layer, 2, 2, scope="pool_1")
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_1_0", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_1_1", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_1_2", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_1_3", name=name)
      layr2 = layer;

      layer = slim.max_pool2d(layer, 2, 2, scope="pool_2")
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_2_0", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_2_1", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_2_2", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_2_3", name=name)
      layr3 = layer;

      layer = slim.max_pool2d(layer, 2, 2, scope="pool_3")
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_3_0", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_3_1", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_3_2", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_3_3", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_3_4", name=name)
      layr4 = layer;

      layer = slim.max_pool2d(layer, 2, 2, scope="pool_4")
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_0", name=name)
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_1", name=name)
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_2", name=name)
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_3", name=name)
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_4", name=name)
      layer = conv2d(layer, 256, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_4_5", name=name)
      layer = slim.conv2d_transpose(layer, 256, 2, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=self.bn, scope="deconv_0");
      layer = tf.concat([layer, layr4], 3)

      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_5_0", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_5_1", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_5_2", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_5_3", name=name)
      layer = conv2d(layer, 128, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_5_4", name=name)
      layer = slim.conv2d_transpose(layer, 128, 2, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=self.bn, scope="deconv_1");
      layer = tf.concat([layer, layr3], 3)

      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_6_0", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_6_1", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_6_2", name=name)
      layer = conv2d(layer, 64, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_6_3", name=name)
      layer = slim.conv2d_transpose(layer, 64, 2, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=self.bn, scope="deconv_2");
      layer = tf.concat([layer, layr2], 3)

      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_7_0", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_7_1", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_7_2", name=name)
      layer = conv2d(layer, 32, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_7_3", name=name)
      layer = slim.conv2d_transpose(layer, 32, 2, 2, padding='VALID', activation_fn=lrelu, normalizer_fn=self.bn, scope="deconv_3");
      layer = tf.concat([layer, layr1], 3)

      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_8_0", name=name)
      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=self.bn, scope="conv_8_1", name=name)
      layer = conv2d(layer, 16, 3, 1, padding='SAME', normalizer_fn=None,    scope="conv_8_2", name=name)

      logits= conv2d(layer,  2, 3, 1, padding='SAME', normalizer_fn=None,    scope="conv_9_0", name=name)
      output = tf.sigmoid(logits, name="sigmoid")
      self.learner_vars = tf.contrib.framework.get_variables(sc)
    return output, logits

