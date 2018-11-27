import os
import sys

from PIL import Image
import numpy as np
from scipy import ndimage

def detection_peaks(im, detection_thres):
    im = (im / 255.0).astype(np.float32)

    # Detecting blobs in the detection image
    dot = ndimage.filters.gaussian_filter(im, 1.0, mode='mirror')
    seg_region = (dot > detection_thres)

    labeled_region, n_region = ndimage.measurements.label(seg_region)

    labeled_im = np.zeros_like(seg_region, dtype=np.uint8)
    for regi in range(1, n_region+1):
        bin_region = (labeled_region == regi)
        x_coors = np.where(bin_region.sum(axis=1))
        y_coors = np.where(bin_region.sum(axis=0))
        minx = np.min(x_coors)
        maxx = np.max(x_coors)
        miny = np.min(y_coors)
        maxy = np.max(y_coors)
        labeled_im[(minx+maxx)//2, (miny+maxy)//2] = 255

    return labeled_im

