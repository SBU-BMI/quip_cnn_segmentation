from PIL import Image
from skimage.color import hed2rgb, rgb2hed
import numpy as np
import glob
import os

for filepath in glob.iglob('image/*.png'):
    img = np.array(Image.open(filepath).convert('RGB')).astype(np.float32)
    adj_add = np.array([[[0.09, 0.09, 0.007]]], dtype=np.float32);
    img = hed2rgb(rgb2hed(img / 255.0) + np.clip(np.random.normal(0, 0.3, (1, 1, 3)), -1, 1) * adj_add) * 255.0

    adj_range = 0.03;
    adj_add = 6;
    rgb_mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32);
    adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (1, 1, 3)).astype(np.float32);
    img = (img - rgb_mean) * adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (1, 1, 3)) * adj_add

    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join('image_augmented', os.path.basename(filepath)))

