import numpy as np
import cv2
import sys
from PIL import Image
from scipy import ndimage, misc
from skimage.color import label2rgb

def extract_a_dataset(patch_id, stride=31, win_size=400):
    with open('a_dataset_and_a_technique/list.txt', 'r') as f:
        ids = f.read().splitlines()

    for id in ids:
        print id
        he_image = misc.imread('a_dataset_and_a_technique/Tissue/{}.png'.format(id))[15:-15, 15:-15, :]
        seg_gt = misc.imread('a_dataset_and_a_technique/Mask/{}.png'.format(id))[15:-15, 15:-15] / 40

        for x in range(0, he_image.shape[0]-win_size, stride) + [he_image.shape[0]-win_size,]:
            for y in range(0, he_image.shape[1]-win_size, stride) + [he_image.shape[1]-win_size,]:
                he = he_image[x:x+win_size, y:y+win_size, :]
                seg = seg_gt[x:x+win_size, y:y+win_size]
                if he.shape[0] != win_size or he.shape[1] != win_size or \
                    seg.shape[0] != win_size or seg.shape[1] != win_size:
                    print 'some shape error', he.shape, seg.shape
                    continue
                misc.imsave('new_data_400x400/image/{}.png'.format(patch_id), he)
                misc.imsave('new_data_400x400/mask/{}.png'.format(patch_id), seg)
                patch_id += 1

    return patch_id

