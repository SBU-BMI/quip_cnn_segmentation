import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

size0 = 400;
size1 = 400;
npatch_per_tile = 3;

def sample_overlap(x, y, fx, fy, xlen, ylen):
    if fx <= x and x <= fx+xlen and fy <= y and y <= fy+ylen:
        return True;
    if fx <= x+xlen and x+xlen <= fx+xlen and fy <= y and y <= fy+ylen:
        return True;
    if fx <= x and x <= fx+xlen and fy <= y+ylen and y+ylen <= fy+ylen:
        return True;
    if fx <= x+xlen and x+xlen <= fx+xlen and fy <= y+ylen and y+ylen <= fy+ylen:
        return True;
    return False;

def sample_xy_fxfy(size0, size1, xlen, ylen):
    x, y, fx, fy = 0, 0, 0, 0;
    while sample_overlap(x, y, fx, fy, xlen, ylen):
        x, y = int(np.random.rand()*size0), int(np.random.rand()*size1);
        fx, fy = int(np.random.rand()*size0), int(np.random.rand()*size1);
    return x, y, fx, fy;

patn = 0;
tile_path = './nuclei_synthesis_40X_online/real_tiles/';
paths = [f for f in listdir(tile_path) if isfile(join(tile_path, f))];

fid = open('output/real_info.txt', 'w');
for path in paths:
    full_tile = np.array(Image.open(join(tile_path, path)).convert('RGB'));
    for n in range(npatch_per_tile):
        x, y, fx, fy = sample_xy_fxfy(full_tile.shape[0]-size0, full_tile.shape[1]-size1, size0, size1);
        patch0 = full_tile[x:x+size0, y:y+size1, :];
        patch1 = full_tile[fx:fx+size0, fy:fy+size1, :];
        Image.fromarray(patch0).save('./output/real/image0_{}.png'.format(patn));
        Image.fromarray(patch1).save('./output/real/image1_{}.png'.format(patn));
        fid.write('{} {}\n'.format(patn, path));
        patn += 1;
fid.close();

