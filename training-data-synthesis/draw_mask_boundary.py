import scipy.stats as st
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from skimage import feature
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

mask_path = './fake_ground_truth/';

def canny_edge(mask, sig=0.1, low_th=10, high_th=40):
    imgray = np.mean(mask, axis = 2);
    edges = feature.canny(imgray, sigma=sig, low_threshold=low_th, high_threshold=high_th);
    return edges;

def canny_edge_on_mask(imgray):
    edges = feature.canny(imgray, sigma=0.1, low_threshold=10, high_threshold=40);
    edges_ret = edges.copy();
    return (edges_ret>0).astype(np.uint8);

paths = [f for f in listdir(mask_path) if isfile(join(mask_path, f))];
for path in paths:
    if len(path.split(os.path.sep)[-1].split('mask_')) == 1:
        continue;
    im_no = path.split(os.path.sep)[-1].split('mask_')[1].split('.png')[0];

    mask = np.array(Image.open(join(mask_path, path)).convert('L'));
    mask_edge = canny_edge_on_mask(((mask[:,:]>0).astype(np.uint8)*255)).astype(np.float32);
    mask_edge = gaussian_filter(mask_edge, sigma=2.5);
    mask_edge = np.clip(mask_edge*255*3.0, 0, 255).astype(np.uint8);

    Image.fromarray(mask_edge).save('{}/maskedge_{}.png'.format(mask_path, im_no));


