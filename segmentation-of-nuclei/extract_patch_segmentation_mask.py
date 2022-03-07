from PIL import Image, ImageDraw
import numpy as np
from glob import glob
from os import path
import csv

def tile_intersect_patch(x0, y0, sx0, sy0, x1, y1, sx1, sy1):
    if x0 > x1 + sx1 or x1 > x0 + sx0:
        return False
    if y0 > y1 + sy1 or y1 > y0 + sy0:
        return False
    return True

def info2patch_xy(segmentation_polygon_folder, x, y, pw):
    fns = []
    mpp = 0.25
    for fn in glob(path.join(segmentation_polygon_folder, '*.csv')):
        # 100001_100001_4000_3162_0.2277_1-features.csv
        px, py, pw_x, pw_y, mpp, _ = path.basename(fn).split('_')
        px = int(px)
        py = int(py)
        pw_x = int(pw_x)
        pw_y = int(pw_y)
        mpp = float(mpp)
        if tile_intersect_patch(px, py, pw_x, pw_y, x, y, pw, pw):
            fns.append(fn)

    return fns, x, y, pw, mpp

def patch_xy2mask(patch_xy, scale_to_40X=True):
    if not patch_xy:
        return None

    poly_paths, x, y, pw, mpp = patch_xy
    scale_f = (mpp * 4) if scale_to_40X else 1.0
    pw = int(pw * scale_f)
    mask = Image.fromarray(np.zeros((pw, pw), dtype=np.int16));
    draw = ImageDraw.Draw(mask);

    nuc_idx = 1
    for poly_path in poly_paths:
        with open(poly_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                coors = [float(n) for n in row["Polygon"][1:-1].split(':')]
                for i in range(0, len(coors), 2):
                    coors[i] = scale_f * (coors[i] - x)
                for i in range(1, len(coors), 2):
                    coors[i] = scale_f * (coors[i] - y)
                coors += [coors[0], coors[1]]
                if min(coors[0::2]) > pw or max(coors[0::2]) < 0 or \
                   min(coors[1::2]) > pw or max(coors[1::2]) < 0:
                    continue

                draw.polygon(coors, fill=(nuc_idx))
                nuc_idx += 1

    return np.array(mask)

def extract_segmentation_mask(
        segmentation_polygon_folder, x, y, patch_width, scale_to_40X=True):
    '''
    Extract segmentation mask

    Args:
        segmentation_polygon_folder: path to a segmentation output folder.
        x: x coordinate of the patch you want to extract.
        y: y coordinate of the patch.
        patch_width: size of the patch.
    Returns:
        Instance-level mask as numpy array.
    '''

    patch_xy = info2patch_xy(
            segmentation_polygon_folder, x, y, patch_width)
    return patch_xy2mask(patch_xy, scale_to_40X)
