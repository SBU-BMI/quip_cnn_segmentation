from scipy import misc
from PIL import Image, ImageDraw
import numpy as np
import sys
import csv
import os

seg_path = sys.argv[1] # path to XXX_SEG.png
out_path = sys.argv[2] # output file path

poly_path = seg_path[:-len('_SEG.png')] + '-features.csv'

fields = os.path.basename(seg_path).split('_')
x_off = float(fields[0])
y_off = float(fields[1])
im = Image.open(seg_path)
draw = ImageDraw.Draw(im)

with open(poly_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        coors = [float(n) for n in row["Polygon"][1:-1].split(':')]
        for i in range(0, len(coors), 2):
            coors[i] -= x_off
        for i in range(1, len(coors), 2):
            coors[i] -= y_off
        coors += [coors[0], coors[1]]
        draw.line(tuple(coors), fill="yellow", width=2)

im.save(out_path)

