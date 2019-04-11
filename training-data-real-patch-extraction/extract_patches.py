import numpy as np
import cv2
import sys
import os
from PIL import Image
from scipy import ndimage, misc
from skimage.color import label2rgb
from skimage import draw
import xml.etree.ElementTree as ET


def poly2mask(vertex_row_coords, vertex_col_coords, mask):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, mask.shape)
    mask[fill_row_coords, fill_col_coords] = 1
    mask[int((min(vertex_row_coords)+max(vertex_row_coords))/2),
         int((min(vertex_col_coords)+max(vertex_col_coords))/2)] += 4
    return mask


patch_id = 1
stride = 31
win_size = 400

if not os.path.isdir('MoNuSeg Training Data/Mask'):
    os.makedirs('MoNuSeg Training Data/Mask')

ids = []
for filename in os.listdir('MoNuSeg Training Data/Annotations/'):
    if not filename.endswith('.xml'):
        continue
    ids.append(filename[:-len('.xml')])


for id in ids:
    print 'converting xml to png. {}'.format(id)

    tissue = Image.open('MoNuSeg Training Data/Tissue images/{}.tif'.format(id))
    s0, s1 = tissue.size[0], tissue.size[1]

    tree = ET.parse('MoNuSeg Training Data/Annotations/{}.xml'.format(id))
    root = tree.getroot()

    mask = np.zeros((s0, s1), dtype=np.uint8)
    for Annotation in root.iter('Annotation'):
        for Regions in Annotation.iter('Regions'):
            for Region in Regions.iter('Region'):
                x_ver = []
                y_ver = []
                for Vertices in Region.iter('Vertices'):
                    for Vertex in Vertices.iter('Vertex'):
                        x_ver.append(float(Vertex.attrib['X']))
                        y_ver.append(float(Vertex.attrib['Y']))
                mask = poly2mask(y_ver, x_ver, mask)

    misc.imsave('MoNuSeg Training Data/Mask/{}.png'.format(id), mask*40)


for id in ids:
    print 'extracting patches from png. {}'.format(id)

    he_image = np.array(Image.open('MoNuSeg Training Data/Tissue images/{}.tif'.format(id)).convert('RGB'))[15:-15, 15:-15, :]
    seg_gt = misc.imread('MoNuSeg Training Data/Mask/{}.png'.format(id))[15:-15, 15:-15] / 40

    for x in range(0, he_image.shape[0]-win_size, stride) + [he_image.shape[0]-win_size,]:
        for y in range(0, he_image.shape[1]-win_size, stride) + [he_image.shape[1]-win_size,]:
            he = he_image[x:x+win_size, y:y+win_size, :]
            seg = seg_gt[x:x+win_size, y:y+win_size]
            if he.shape[0] != win_size or he.shape[1] != win_size or \
                seg.shape[0] != win_size or seg.shape[1] != win_size:
                print 'some shape error', he.shape, seg.shape
                continue
            misc.imsave('real_data_400x400/image_sup/{}.png'.format(patch_id), he)
            misc.imsave('real_data_400x400/mask_sup/{}.png'.format(patch_id), seg)
            patch_id += 1

