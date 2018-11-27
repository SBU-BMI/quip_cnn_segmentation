from skimage import draw
from scipy import ndimage, misc
from PIL import Image
import numpy as np

def read_instance(im_file, rf):
    file_id = im_file[: -len('.png')]

    he_image = np.array(Image.open('{}.png'.format(file_id)).convert('RGB')).astype(np.uint8)
    s0, s1 = he_image.shape[0], he_image.shape[1]

    f = open('{}_data.txt'.format(file_id), 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()

    mask = np.zeros((s0, s1), dtype=np.uint8)
    detect = np.zeros((int(s0*rf), int(s1*rf)), dtype=np.uint8)
    for n, line in enumerate(lines):
        xy = line.split()
        ys = [float(coor) for coor in xy[0::2]]
        xs = [float(coor) for coor in xy[1::2]]
        fill_row, fill_col = draw.polygon(xs, ys, mask.shape)
        mask[fill_row, fill_col] = 1
        detect[int(rf*(min(xs)+max(xs))/2), int(rf*(min(ys)+max(ys))/2)] = 4

    he_image = misc.imresize(he_image, (int(s0*rf), int(s1*rf)), interp='bilinear')
    mask = misc.imresize(mask, (int(s0*rf), int(s1*rf)), interp='nearest')

    return he_image, mask + detect

def read_folder(in_path):
    he_image_list = []
    seg_ground_truth_list = []

    f = open('{}/image_resize_list.txt'.format(in_path), 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
    for n, line in enumerate(lines):
        im_name = '{}/{}'.format(in_path, line.split()[0])
        resize_factor = float(line.split()[1])

        for rf in [0.85*resize_factor, resize_factor, 1.15*resize_factor]:
            he_image, seg_ground_truth = read_instance(im_name, rf)
            he_image_list.append(he_image)
            seg_ground_truth_list.append(seg_ground_truth)

    return he_image_list, seg_ground_truth_list

