import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

from utils import imread, imwrite

DATA_FNAME = 'real.npz'
REAL_CROP_S = 32;
REAL_CROP_E = 107;

def imread_rgb_image(path):
  return np.array(Image.open(path).convert('RGB'));

def maybe_preprocess(config, data_path, sample_path=None):
  if config.max_synthetic_num < 0:
    max_synthetic_num = None
  else:
    max_synthetic_num = config.max_synthetic_num

  # Nuclei dataset
  base_path = os.path.join(data_path, config.real_image_dir)
  npz_path = os.path.join(data_path, DATA_FNAME)

  if not os.path.exists(npz_path):
    png_paths = []
    for root, dirnames, filenames in os.walk(base_path):
      for filename in fnmatch.filter(filenames, 'image0_*.png'):
        png_paths.append(os.path.join(root, filename))

    print("[*] Preprocessing real `nuclei` data...")

    real_images = []
    ref_real_images = []
    for png_path in tqdm(png_paths):
      png0_path = png_path;
      png0 = imread_rgb_image(png0_path);
      real_images.extend(png0[np.newaxis,
              REAL_CROP_S:REAL_CROP_E, REAL_CROP_S:REAL_CROP_E, :]);
      png1_path = '/'.join(png_path.split('/')[0:-1]) \
              + '/image1_' + png_path.split('_')[-1];
      png1 = imread_rgb_image(png1_path);
      ref_real_images.extend(png1[np.newaxis,
              REAL_CROP_S:REAL_CROP_E, REAL_CROP_S:REAL_CROP_E, :]);

    real_data = np.stack(real_images, axis=0)
    ref_real_data = np.stack(ref_real_images, axis=0)
    np.savez(npz_path, real=real_data, ref_real=ref_real_data)

  # Fake-images dataset
  synthetic_image_path = os.path.join(data_path, config.synthetic_image_dir);
  png_paths = glob(os.path.join(synthetic_image_path, '*.png'))
  print("[*] # of synthetic data: {}".format(len(png_paths)))
  print("[*] Finished preprocessing synthetic `nuclei` data.")

  return synthetic_image_path

def load(config, data_path, sample_path, rng):
  if not os.path.exists(data_path):
    print('creating folder', data_path)
    os.makedirs(data_path)

  synthetic_image_path = maybe_preprocess(config, data_path, sample_path)

  nuclei_data = np.load(os.path.join(data_path, DATA_FNAME))
  real_data = nuclei_data['real']
  ref_real_data = nuclei_data['ref_real']

  if not os.path.exists(sample_path):
    os.makedirs(sample_path)

  print("[*] Save samples images in {}".format(data_path))
  random_idxs = rng.choice(len(real_data), 100)
  for idx, random_idx in enumerate(random_idxs):
    image_path = os.path.join(sample_path, "real_{}.png".format(idx))
    imwrite(image_path, real_data[random_idx])

  return real_data, synthetic_image_path, ref_real_data

class DataLoader(object):
  def __init__(self, config, rng=None):
    self.rng = np.random.RandomState(1) if rng is None else rng

    self.data_path = os.path.join(config.data_dir, 'nuclei')
    self.sample_path = os.path.join(self.data_path, config.sample_dir)
    self.batch_size = config.batch_size
    self.debug = config.debug

    self.real_data, synthetic_image_path, self.ref_real_data = load(config, self.data_path, self.sample_path, rng)

    self.synthetic_data_paths = np.array(glob(os.path.join(synthetic_image_path, '*.png')))
    self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0]).shape);
    self.synthetic_data_paths.sort()

    if self.real_data.ndim == 3:
      self.real_data = np.expand_dims(self.real_data, -1)

    self.real_p = 0

  def get_observation_size(self):
    return self.real_data.shape[1:]

  def reset(self):
    self.real_p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    if self.real_p == 0:
      inds = self.rng.permutation(self.real_data.shape[0])
      self.real_data = self.real_data[inds]
      self.ref_real_data = self.ref_real_data

    if self.real_p + n > self.real_data.shape[0]:
      self.reset()

    x = self.real_data[self.real_p : self.real_p + n]
    y = self.ref_real_data[self.real_p : self.real_p + n]
    self.real_p += self.batch_size

    if np.random.rand(1) < 0.5:
        return x, y
    else:
        return y, x

  next = __next__

