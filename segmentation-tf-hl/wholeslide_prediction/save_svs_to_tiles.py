import numpy as np
import h5py
import sys
import os
from PIL import Image

#arg1: slide name
#arg2: input folder
#arg3: ouput folder

slide_name =  sys.argv[1];
slide_file_name=sys.argv[1].split('/')
slide_file_name=slide_file_name[len(slide_file_name)-1]
output_folder = sys.argv[2] + '/' + slide_file_name;
tile_size = 4000;

if not os.path.exists(output_folder):
    os.mkdir(output_folder);

try:
    mpp_w_h = os.popen('bash get_mpp_w_h.sh {}'.format(slide_name)).read();
    if len(mpp_w_h.split()) != 3:
        print '{}: mpp_w_h wrong'.format(slide_name);
        exit(1);

    mpp = float(mpp_w_h.split()[0]);
    width = int(mpp_w_h.split()[1]);
    height = int(mpp_w_h.split()[2]);
    if (mpp < 0.01 or width < 1 or height < 1):
        print '{}: mpp, width, height wrong'.format(slide_name);
        exit(1);
except:
    print '{}: exception caught'.format(slide_name);
    exit(1);

print slide_name, width, height;

for x in range(1, width, tile_size):
    for y in range(1, height, tile_size):
        
        if x + tile_size >= width:
            pw_x = width - x - 1;
        else:
            pw_x = tile_size;
        if y + tile_size >= height:
            pw_y = height - y - 1;
        else:
            pw_y = tile_size;
        fname = '{}/{}_{}_{}_{}_{}.png'.format(output_folder, x, y, pw_x, pw_y, mpp);
        print(fname)
        if os.path.isfile(output_folder+'/'+str(x)+'_'+ str(y)+'_'+ str(pw_x)+'_'+ str(pw_y)+'_'+ str(mpp)+'.png'):
          print(str(x)+'_'+ str(y)+'_'+ str(pw_x)+'_'+ str(pw_y)+'_'+ str(mpp))
          continue       
        os.system('bash save_tile.sh {} {} {} {} {} {}'.format(slide_name, x, y, pw_x, pw_y, fname));

