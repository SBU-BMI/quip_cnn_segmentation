"""Segmentation results generated using code prior to Dec. 7th, 2019
have a bug: segmentation results (if there are any) close to the
boundary of a WSI are mis-aligned with the actual image.

Please use this script to fix the segmentation results.

Usage:
  python one_pass_fix.py segmentation_output_folder

  The segmentation_output_folder should contain polygon
  files (*-features.csv) and gray-scale nucleus probability maps (*_SEG.png)
"""

import sys
import os
import csv
from PIL import Image
from glob import glob

def fix_polygon_csv(in_folder):
    fns = glob('{}/*-features.csv'.format(in_folder))

    for fn in fns:
        x_off, y_off, w, h, mpp, _ = os.path.basename(fn).split('_')
        if w == h:
            continue
        x_off = int(x_off)
        y_off = int(y_off)
        w = int(w)
        h = int(h)
        x_fixer = float(w) / h
        y_fixer = float(h) / w

        fn_fix_tmp = fn + '.fix.temp'
        fid = open(fn_fix_tmp, 'w')
        fid.write('AreaInPixels,PhysicalSize,Polygon\n')

        with open(fn, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                area_pixel = row['AreaInPixels']
                physical_size = row['PhysicalSize']
                poly = [float(n) for n in row["Polygon"][1:-1].split(':')]
                for i in range(0, len(poly), 2):
                    poly[i] -= x_off
                    poly[i] *= x_fixer
                    poly[i] += x_off
                for i in range(1, len(poly), 2):
                    poly[i] -= y_off
                    poly[i] *= y_fixer
                    poly[i] += y_off
                poly += [poly[0], poly[1]]
                poly_str = ':'.join(['{:.1f}'.format(x) for x in poly])
                fid.write('{},{},[{}]\n'.format(area_pixel, physical_size, poly_str))

        fid.close()
        os.rename(fn_fix_tmp, fn)

def fix_probability_map_png(in_folder):
    fns = glob('{}/*_SEG.png'.format(in_folder))

    for fn in fns:
        x_off, y_off, w, h, mpp, _, _ = os.path.basename(fn).split('_')
        if w == h:
            continue
        w = int(w)
        h = int(h)

        im = Image.open(fn).resize(size=(w, h), resample=Image.BILINEAR)
        im.save(fn)

if len(sys.argv) != 2:
    print(('Usage: python {} segmentation_folder'.format(sys.argv[0])))
    exit(1)
in_folder = sys.argv[1]
fix_polygon_csv(in_folder)
fix_probability_map_png(in_folder)
