from scipy import ndimage
from PIL import Image
import numpy as np

def read_instance(im_file, rf):
    file_id = im_file[: -len('.png')]

    f = open('{}_mask.txt'.format(file_id), 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()

    for n, line in enumerate(lines):
        if n == 0:
            patch_H = int(line.split()[0])
            patch_W = int(line.split()[1])
            seg_ground_truth = np.zeros((patch_H * patch_W, )).astype(np.uint32)
            continue
        seg_ground_truth[n-1] = int(line)
    seg_ground_truth = np.array(Image.fromarray(
        seg_ground_truth.reshape(patch_H, patch_W)).resize(
        (int(patch_H*rf), int(patch_W*rf)), resample=Image.NEAREST))

    det_ground_truth = np.zeros_like(seg_ground_truth, dtype=np.uint8)
    for regi in range(1, seg_ground_truth.max()+1):
        bin_region = (seg_ground_truth == regi)
        if bin_region.sum() == 0:
            continue
        minx = np.min(np.where(bin_region.sum(axis=1)))
        maxx = np.max(np.where(bin_region.sum(axis=1)))
        miny = np.min(np.where(bin_region.sum(axis=0)))
        maxy = np.max(np.where(bin_region.sum(axis=0)))
        det_ground_truth[(minx+maxx)//2, (miny+maxy)//2] = 255

    he_image = np.array(Image.open('{}.png'.format(file_id)).convert('RGB').resize(
        (int(patch_H*rf), int(patch_W*rf)), resample=Image.BILINEAR))

    return he_image, seg_ground_truth, det_ground_truth

def read_folder(in_path):
    he_image_list = []
    seg_ground_truth_list = []
    det_ground_truth_list = []

    f = open('{}/image_resize_list.txt'.format(in_path), 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
    for n, line in enumerate(lines):
        im_name = '{}/{}'.format(in_path, line.split()[0])
        resize_factor = float(line.split()[1])

        for rf in [0.85*resize_factor, resize_factor, 1.15*resize_factor]:
            he_image, seg_ground_truth, det_ground_truth = read_instance(im_name, rf)
            he_image_list.append(he_image)
            seg_ground_truth_list.append(seg_ground_truth)
            det_ground_truth_list.append(det_ground_truth)

    return he_image_list, seg_ground_truth_list, det_ground_truth_list

