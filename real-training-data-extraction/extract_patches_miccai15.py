import read_file_miccai15
import numpy as np
import cv2
import sys
from PIL import Image
from scipy import ndimage, misc
from skimage.color import label2rgb

def extract_miccai15(patch_id, stride=31, win_size=400):
    he_image_list, seg_ground_truth_list = \
        read_file_miccai15.read_folder('./miccai_15/')

    for he_image, seg_ground_truth in zip(he_image_list, seg_ground_truth_list):
        print he_image.shape, seg_ground_truth.shape

        for x in range(0, he_image.shape[0]-win_size, stride) + [he_image.shape[0]-win_size,]:
            for y in range(0, he_image.shape[1]-win_size, stride) + [he_image.shape[1]-win_size,]:
                he = he_image[x:x+win_size, y:y+win_size, :]
                seg = seg_ground_truth[x:x+win_size, y:y+win_size]
                if he.shape[0] != win_size or he.shape[1] != win_size or \
                    seg.shape[0] != win_size or seg.shape[1] != win_size:
                    print 'some shape error', he.shape, seg.shape
                    continue
                misc.imsave('new_data_400x400/image/{}.png'.format(patch_id), he)
                misc.imsave('new_data_400x400/mask/{}.png'.format(patch_id), seg)
                patch_id += 1

    return patch_id
