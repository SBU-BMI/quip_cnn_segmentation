from os import path, makedirs
from glob import glob
from PIL import Image, ImageDraw
import numpy as np
import math
import csv
from multiprocessing import Pool


def path_root(directory):
    head, tail = path.split(directory)
    while head != '' and head != '/':
        head, tail = path.split(head)
    return tail

def path_slide(directory):
    head, _ = path.split(directory)
    return path.basename(head)

def path_tile(directory):
    return path.basename(directory)


def path_mkdir(directory):
    if not path.isdir(directory):
        try:
            makedirs(directory)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


def idx_to_rgb(idx):
    idx = idx * 127
    r = idx / 230 / 230 % 230 + 25
    g = idx / 230 % 230 + 25
    b = idx % 230 + 25
    return (r, g, b)


def process_poly(poly_path):
    try:
        root = path_root(poly_path)
        slide = path_slide(poly_path)
        cancer_type, _ = root.split('_')
        tile = path_tile(poly_path)
        output_tile = tile[:-len('-features.csv')] + '-mask.png'

        output_path = path.join(cancer_type + '_mask', slide, output_tile)

        fields = path.basename(tile).split('_')
        x_off = int(fields[0])
        y_off = int(fields[1])
        width = int(fields[2])
        height = int(fields[3])

        mask = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(mask)
        nuc_idx = 1
        with open(poly_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                coors = [float(n) for n in row["Polygon"][1:-1].split(':')]
                for i in range(0, len(coors), 2):
                    coors[i] -= x_off
                for i in range(1, len(coors), 2):
                    coors[i] -= y_off
                coors += [coors[0], coors[1]]
                draw.polygon(coors, fill=idx_to_rgb(nuc_idx))
                nuc_idx += 1

        path_mkdir(path.join(cancer_type + '_mask', slide))
        mask.save(output_path)
    except:
        print('exception for', poly_path)

poly_paths = glob('*_polygon/*/*-features.csv')

p = Pool(10)
p.map(process_poly, poly_paths)

