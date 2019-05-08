import numpy as np
import openslide
import sys
import os
from PIL import Image
from color_norm.color_normalize import reinhard_normalizer

def white_ratio(pat):
    white_count = 0.0
    total_count = 0.001
    for x in range(0, pat.shape[0]-200, 100):
        for y in range(0, pat.shape[1]-200, 100):
            p = pat[x:x+200, y:y+200, :]
            whiteness = (np.std(p[:,:,0]) + np.std(p[:,:,1]) + np.std(p[:,:,2])) / 3.0
            if whiteness < 14:
                white_count += 1.0
            total_count += 1.0
    return white_count/total_count


def stain_normalized_tiling(slide_name, patch_size, do_actually_read_image=True):
    margin = 5
    try:
        oslide = openslide.OpenSlide(slide_name)
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        elif "XResolution" in oslide.properties:
            mpp = float(oslide.properties["XResolution"]);
        elif 'tiff.XResolution' in oslide.properties:
            mpp = float(oslide.properties["tiff.XResolution"]);
        else:
            mpp = 0.250

        if mpp < 0.375:
            scale_factor = 1
        else:
            scale_factor = 2
        pw = patch_size
        width = oslide.dimensions[0]
        height = oslide.dimensions[1]
    except:
        print 'Error in {}: exception caught exiting'.format(slide_name)
        raise Exception('{}: exception caught exiting'.format(slide_name))
        return

    n40X = reinhard_normalizer('color_norm/target_40X.png')

    for x in range(1, width, pw):
        for y in range(1, height, pw):
            if x + pw > width - margin:
                pw_x = width - x - margin
            else:
                pw_x = pw
            if y + pw > height - margin:
                pw_y = height - y - margin
            else:
                pw_y = pw

            if pw_x <= 0 or pw_y <= 0:
                continue

            if do_actually_read_image:
                try:
                    patch = oslide.read_region((x, y), 0, (pw_x, pw_y)).convert('RGB')
                except:
                    print '{}: exception caught'.format(slide_name)
                    continue
            else:
                patch = Image.new('RGB', (pw_x, pw_y), (255, 255, 255))

            ori_size0 = patch.size[0]
            ori_size1 = patch.size[1]
            patch = np.array(patch.resize(
                (patch.size[0]*scale_factor, patch.size[1]*scale_factor), Image.ANTIALIAS))

            if white_ratio(patch) < 0.25:
                patch = n40X.normalize(patch)

            yield patch, (x, y, pw_x, pw_y, ori_size0, ori_size1, mpp, scale_factor), (width, height)

