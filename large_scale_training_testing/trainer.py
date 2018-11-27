import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import scipy.stats as st
import glob
from scipy import misc
from PIL import Image
from skimage import color
from scipy.misc import imresize
from layers import normalize
import sys
import glob
import random
from glob import iglob

from model import Model
from buffer import Buffer
import data.nuclei_data as nuclei_data
from utils import imwrite, imread, img_tile, synthetic_to_refer_paths
from preprocess import stain_normalized_tiling
from postprocess import MultiProcWatershed

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step_d_g = config.max_step_d_g
    self.max_step_d_g_l = config.max_step_d_g_l

    if not config.is_train:
      config.input_width = config.input_PS_test
      config.input_height = config.input_PS_test

    self.load_path = config.load_path
    self.seg_path = config.seg_path
    self.out_path = config.out_path
    self.K_d = config.K_d
    self.K_g = config.K_g
    self.K_l = config.K_l
    self.initial_K_d = config.initial_K_d
    self.initial_K_g = config.initial_K_g
    self.initial_K_l = config.initial_K_l
    self.after_K_l = config.after_K_l
    self.checkpoint_secs = config.checkpoint_secs

    self.method_description = config.method_description
    self.postprocess_nproc = config.postprocess_nproc
    self.postprocess_seg_thres = config.postprocess_seg_thres
    self.postprocess_det_thres = config.postprocess_det_thres
    self.postprocess_win_size = config.postprocess_win_size
    self.postprocess_min_nucleus_size = config.postprocess_min_nucleus_size
    self.postprocess_max_nucleus_size = config.postprocess_max_nucleus_size
    self.only_postprocess = config.only_postprocess

    DataLoader = {
        'nuclei': nuclei_data.DataLoader,
    }[config.data_set]
    self.data_loader = DataLoader(config, rng=self.rng)

    self.model = Model(config, self.data_loader)
    if config.is_train:
      self.history_buffer = Buffer(config, self.rng)

    self.summary_ops = {
        'test_synthetic_images': {
            'summary': tf.summary.image("test_synthetic_images",
                                        self.model.x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.x,
        },
        'test_refined_images': {
            'summary': tf.summary.image("test_refined_images",
                                        self.model.denormalized_R_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.denormalized_R_x,
        },
        'test_refer_images': {
            'summary': tf.summary.image("test_refer_images",
                                        self.model.ref_image,
                                        max_outputs=config.max_image_summary),
            'output': self.model.ref_image,
        },
        'test_learner_outputs': {
            'summary': tf.summary.image("test_learner_outputs",
                                        self.model.L_R_x[..., 0:1]*255,
                                        max_outputs=config.max_image_summary),
            'output': self.model.L_R_x[..., 0:1]*255,
        },
    }

    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.learner_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    print("[*] Training starts...")
    self._summary_writer = None

    #sample_num = reduce(lambda x, y: x*y, self.config.sample_image_grid)
    #idxs = self.rng.choice(len(self.data_loader.synthetic_data_paths), sample_num)

    sample_x = self.config.sample_image_grid[0]
    sample_y = self.config.sample_image_grid[1]
    idxs = self.rng.choice(len(self.data_loader.synthetic_data_paths), sample_x)
    synthetic_paths = self.data_loader.synthetic_data_paths[idxs];
    synthetic_ref_paths = synthetic_to_refer_paths(synthetic_paths, self.config);

    synthetic_paths = np.tile(synthetic_paths, [sample_y, 1]).flatten('F')
    synthetic_ref_paths = np.tile(synthetic_ref_paths, [sample_y, 1]).flatten('C')

    test_samples = np.stack([imread(path) for path in synthetic_paths]);
    test_refer_samples = np.stack([imread(path) for path in synthetic_ref_paths]);

    if test_samples.ndim == 3:
      test_samples = np.expand_dims(test_samples, -1)
    test_samples = test_samples[:, 0:self.config.input_height, 0:self.config.input_width, :];
    if test_refer_samples.ndim == 3:
      test_refer_samples = np.expand_dims(test_refer_samples, -1)
    test_refer_samples = test_refer_samples[:, 0:self.config.input_height, 0:self.config.input_width, :];

    def train_refiner(push_buffer=False):
      feed_dict = {
        self.model.synthetic_batch_size: self.data_loader.batch_size,
      }
      res = self.model.train_refiner(
          self.sess, feed_dict, self._summary_writer, with_output=True)
      self._summary_writer = self._get_summary_writer(res)

      if push_buffer:
        self.history_buffer.push(res['output'])

      if res['step'] % self.log_step == 0:
        feed_dict = {
            self.model.x: test_samples,
            self.model.ref_image: test_refer_samples,
        }
        self._inject_summary(
          'test_refined_images', feed_dict, res['step'])
        self._inject_summary(
          'test_learner_outputs', feed_dict, res['step'])

        if res['step'] / float(self.log_step) == 1.:
          self._inject_summary(
              'test_synthetic_images', feed_dict, res['step'])
          self._inject_summary(
              'test_refer_images', feed_dict, res['step'])

    def train_discrim():
      a, b, c = self.history_buffer.sample()
      d, e = self.data_loader.next()

      feed_dict = {
        self.model.synthetic_batch_size: self.data_loader.batch_size/2,
        self.model.R_x_history: a,
        self.model.refimg_history: c,
        self.model.y: d,
        self.model.ref_y: e,
      }
      res = self.model.train_discrim(
          self.sess, feed_dict, self._summary_writer, with_history=True, with_output=False)
      self._summary_writer = self._get_summary_writer(res)

    def train_learner():
      a, b, c = self.history_buffer.sample()

      feed_dict = {
        self.model.synthetic_batch_size: self.data_loader.batch_size/2,
        self.model.R_x_history: a,
        self.model.mask_history: b,
        self.model.refimg_history: c,
      }
      res = self.model.train_learner(
          self.sess, feed_dict, self._summary_writer, with_output=False)

      self._summary_writer = self._get_summary_writer(res)

    for k in trange(self.initial_K_g, desc="Train refiner"):
      train_refiner(push_buffer=(k>self.initial_K_g*0.9))

    for k in trange(self.initial_K_d, desc="Train discrim"):
      train_discrim()

    for step in trange(self.max_step_d_g, desc="Train refiner+discrim"):
      for k in xrange(self.K_g):
        train_refiner(push_buffer=True)

      for k in xrange(self.K_d):
        train_discrim()

    for k in trange(self.initial_K_l, desc="Train learner"):
      train_learner()

    for step in trange(self.max_step_d_g_l, desc="Train all Three"):
      for k in xrange(self.K_g):
        train_refiner(push_buffer=True)
      for k in xrange(self.K_l):
        train_learner()
      for k in xrange(self.K_d):
        train_discrim()

    for k in trange(self.after_K_l, desc="Train learner"):
      train_learner()

  def test(self):
    def get_image_id(svs_path, delimitor='.'):
      return os.path.basename(svs_path).split(delimitor)[0]

    svs_list = [(f, get_image_id(f)) for f in iglob(self.seg_path + '/*.svs') if os.path.isfile(f)]
    tif_list = [(f, get_image_id(f)) for f in iglob(self.seg_path + '/*.tif') if os.path.isfile(f)]

    self.watershed_manager = MultiProcWatershed(n_proc=self.postprocess_nproc)
    self.watershed_manager.start()

    for wsi, image_id in (svs_list + tif_list):
      try:
        self.cnn_pred_mask(wsi, image_id)
      except:
        print 'Segmentation failed for {}'.format(wsi)
        continue

    self.watershed_manager.wait_til_stop()

  def cnn_pred_mask(self, wsi_path, image_id):
    PS = self.config.input_PS_test
    step_size = self.config.pred_step_size
    gsm = np.ones((PS, PS, 1), dtype=np.float32) * 1e-6
    gsm[1:-1, 1:-1, 0] = 0.01;
    gsm[50:-50, 50:-50, 0] = 1;
    batch_size = self.config.pred_batch_size;

    outfolder = os.path.join(self.out_path, os.path.basename(wsi_path))
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    for uint8patch, patch_info, wsi_dim in stain_normalized_tiling(
            wsi_path, 4000, self.only_postprocess):
      patch = normalize(uint8patch.astype(np.float32))
      px, py, pw_x, pw_y, ori_size0, ori_size1, mpp, scale_factor = patch_info
      outf = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_SEG.png'.format(
                                     px, py, pw_x, pw_y, mpp, scale_factor))

      # Check if patch is too small to handle
      if patch.shape[0] < PS or patch.shape[1] < PS:
        continue;

      # Check if skip the CNN step
      if not self.only_postprocess:
        print "CNN segmentation on", outf

        pred_m = np.zeros((patch.shape[0], patch.shape[1], 3), dtype=np.float32);
        num_m = np.zeros((patch.shape[0], patch.shape[1], 1), dtype=np.float32) + 4e-6;

        net_inputs = [];
        xy_indices = [];
        for x in range(0, pred_m.shape[0]-PS+1, step_size) + [pred_m.shape[0]-PS,]:
          for y in range(0, pred_m.shape[1]-PS+1, step_size) + [pred_m.shape[1]-PS,]:
            pat = patch[x:x+PS, y:y+PS, :]
            wh = pat[...,0].std() + pat[...,1].std() + pat[...,2].std();
            if wh >= 0.18:
              net_inputs.append(pat.transpose());
              xy_indices.append((x, y));

            if len(net_inputs)>=batch_size or (x==pred_m.shape[0]-PS and y==pred_m.shape[1]-PS and len(net_inputs)>0):
              feed_dict = {
                self.model.test_patch_normalized: np.transpose(np.array(net_inputs), [0,2,3,1]),
              }
              res_discrim = self.model.test_learner_patch(self.sess, feed_dict, None, with_output=True);
              net_outputs = res_discrim['output']
              net_outputs = np.concatenate((net_outputs, np.zeros((
                  net_outputs.shape[0], net_outputs.shape[1], net_outputs.shape[2], 1), dtype=np.float32)), axis=-1)
              net_outputs = np.swapaxes(net_outputs, 1, 2)
              for outi, (x, y) in enumerate(xy_indices):
                pred_m[x:x+PS, y:y+PS, :] += net_outputs[outi, ...] * gsm;
                num_m[x:x+PS, y:y+PS, :] += gsm;
              net_inputs = [];
              xy_indices = [];

        pred_m /= num_m;
        pred_m = misc.imresize((pred_m*255).astype(np.uint8), (ori_size0, ori_size1));
        imwrite(outf, pred_m);
        #he_outf = os.path.join(outfolder, '{}_{}_{}_{}_{}_{}_HE.png'.format(
        #                                  px, py, pw_x, pw_y, mpp, scale_factor))
        #imwrite(he_outf, uint8patch);

      if os.path.isfile(outf):
        watershed_params = {
              'in_path': outf,
              'image_id': image_id,
              'wsi_width': wsi_dim[0],
              'wsi_height': wsi_dim[1],
              'method_description': self.method_description,
              'seg_thres': self.postprocess_seg_thres,
              'det_thres': self.postprocess_det_thres,
              'win_size': self.postprocess_win_size,
              'min_nucleus_size': self.postprocess_min_nucleus_size,
              'max_nucleus_size': self.postprocess_max_nucleus_size,
              }
        self.watershed_manager.add_job(watershed_params)

  def _inject_summary(self, tag, feed_dict, step):
    summaries = self.sess.run(self.summary_ops[tag], feed_dict)
    self.summary_writer.add_summary(summaries['summary'], step)

    path = os.path.join(self.config.sample_model_dir, "{}_{}.png".format(tag, step))
    tile = img_tile(summaries['output'], tile_shape=self.config.sample_image_grid);
    if tile.shape[2] == 1:
        tile = tile[:, :, 0];
    imwrite(path, tile);

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
