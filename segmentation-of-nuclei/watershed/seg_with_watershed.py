import numpy as np
import os
import sys
import cv2
from PIL import Image
from scipy import ndimage
from skimage.morphology import watershed
from skimage.color import label2rgb
import time

import detection_binarize
from gen_json import gen_meta_json

def apply_segmentation(in_path, image_id, wsi_width, wsi_height, method_description,
        seg_thres=0.33, det_thres=0.07, win_size=200, min_nucleus_size=20, max_nucleus_size=65536):

    def zero_padding(array, margin):
        size1, size2 = array.shape
        padded = np.zeros((size1+margin*2, size2+margin*2), dtype=array.dtype)
        padded[margin:-margin, margin:-margin] = array
        return padded

    def remove_padding(array, margin):
        return array[margin:-margin, margin:-margin]

    def get_bound_boxes(labeled_region, n_region):
        bbox_arr = [None,] * (n_region + 1)
        if n_region == 0:
            return bbox_arr

        it = np.nditer(labeled_region, flags=['multi_index'])
        while not it.finished:
            v = it[0]
            if v < 1:
                it.iternext()
                continue
            x, y = it.multi_index
            if bbox_arr[v] is None:
                bbox_arr[v] = (x, x, y, y)
            else:
                minx, maxx, miny, maxy = bbox_arr[v]
                bbox_arr[v] = (min(x, minx), max(x, maxx),
                               min(y, miny), max(y, maxy))
            it.iternext()
        return bbox_arr

    def seed_recall(seeds, seg_region, potential):
        labeled_region, n_region = ndimage.measurements.label(seg_region)
        bbox_arr = get_bound_boxes(labeled_region, n_region)

        for regi in range(1, n_region+1):
            minx, maxx, miny, maxy = bbox_arr[regi]
            bin_region = (labeled_region[minx:maxx+1, miny:maxy+1] == regi)
            seed_region = seeds[minx:maxx+1, miny:maxy+1]
            potential_region = potential[minx:maxx+1, miny:maxy+1]

            if (seed_region * bin_region).sum() == 0 and bin_region.sum() < max_nucleus_size:
                potential_local = potential_region * bin_region
                x_offset, y_offset = np.unravel_index(
                        np.argmax(potential_local, axis=None), potential_local.shape)
                seeds[minx+x_offset, miny+y_offset] = True
        return seeds

    def read_instance(im_file, resize_factor):
        det_seg = Image.open(im_file).convert('RGB')
        if resize_factor != 1:
            det_seg = det_seg.resize((det_seg.size[0]*resize_factor,
                det_seg.size[1]*resize_factor), resample=Image.NEAREST)
        det_seg = np.array(det_seg)

        return det_seg[..., 0], det_seg[..., 1]

    print "Watershed postprocessing on", in_path

    file_id = os.path.basename(in_path)[:-len('_SEG.png')]
    resize_factor = int(file_id.split('_')[5])
    detection, segmentation = read_instance(in_path, resize_factor)

    global_xy_offset = [int(x) for x in file_id.split('_')[0:2]]

    time0 = time.time()

    # Padding and smoothing
    padding_size = win_size + 10
    detection = zero_padding(detection_binarize.detection_peaks(detection, det_thres), padding_size)
    segmentation = zero_padding(segmentation, padding_size)
    segmentation = ndimage.filters.gaussian_filter(segmentation, 0.5, mode='mirror')

    seeds = seed_recall(detection>0, segmentation>(seg_thres*255), segmentation)

    markers = ndimage.measurements.label(ndimage.morphology.binary_dilation(seeds, np.ones((3,3))))[0]

    time1 = time.time()
    water_segmentation = watershed(-segmentation, markers,
            mask=(segmentation>(seg_thres*255)), compactness=1.0)

    time2 = time.time()

    xs, ys = np.where(seeds)
    fid = open(os.path.join(os.path.dirname(in_path), file_id+'-features.csv'), 'w')
    fid.write('AreaInPixels,PhysicalSize,Polygon\n')
    for nucleus_id, (x, y) in enumerate(zip(xs, ys)):
        seg_win = water_segmentation[x-win_size:x+win_size+1, y-win_size:y+win_size+1]
        bin_win = (seg_win == water_segmentation[x, y])

        # Fill holes in object
        bin_win = ndimage.binary_fill_holes(bin_win)
        physical_size = bin_win.sum()
        if physical_size < min_nucleus_size or physical_size >= bin_win.size:
            continue

        xoff = float(y - win_size - padding_size)
        yoff = float(x - win_size - padding_size)
        poly = cv2.findContours(bin_win.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2][0][:,0,:].astype(np.float32)
        poly[:, 0] = (poly[:, 0] + xoff) / resize_factor + global_xy_offset[0]
        poly[:, 1] = (poly[:, 1] + yoff) / resize_factor + global_xy_offset[1]
        poly_str = ':'.join(['{:.1f}'.format(x) for x in poly.flatten().tolist()])
        fid.write('{},{},[{}]\n'.format(
            int(physical_size/resize_factor/resize_factor), int(physical_size), poly_str))
    fid.close()

    time3 = time.time()

    print "Time in watershed for ",in_path," Padding/Seed: ",(time1-time0)," Watershed: ",(time2-time1)," Fillholes: ",(time3-time2)

    gen_meta_json(in_path, image_id, wsi_width, wsi_height, method_description,
            seg_thres, det_thres, win_size, min_nucleus_size, max_nucleus_size)
