import numpy as np
import h5py
import sys
import os
from PIL import Image
import subprocess
import concurrent.futures


slide_name =  sys.argv[1];
slide_file_name=sys.argv[1].split('/')
slide_file_name=slide_file_name[len(slide_file_name)-1]
output_folder = sys.argv[2] + '/' + slide_file_name;

tile_size = 4000;
max_workers=8; 

if not os.path.exists(output_folder):
    os.makedirs(output_folder);

try:	
    mpp_w_h = os.popen('bash get_mpp_w_h.sh {}'.format(slide_name)).read();
    print(mpp_w_h.split())
    #if len(mpp_w_h.split()) != 3:
        #print '{}: mpp_w_h wrong'.format(slide_name);
        #exit(1);

    mpp = 0.25
    #float(mpp_w_h.split()[0]);
    width = int(mpp_w_h.split()[0]);
    height = int(mpp_w_h.split()[1]);
    if (mpp < 0.01 or width < 1 or height < 1):
        print '{}: mpp, width, height wrong'.format(slide_name);
        exit(1);
except:
    print '{}: exception caught'.format(slide_name);
    exit(1);

print slide_name, width, height;

count=0
count2=[]
count3=[]

###############################################
def process_one_tile(x,y):
  if width-x<=80 or height-y<=80:
    if width-x<=80:
      count3.append(width-x)
    if height-y<=80:
      count2.append(height-y)
  else:
    if x + tile_size +80 >= width:
      pw_x = width - x - 1;
    else:
      pw_x = tile_size;
     
    if y + tile_size +80>= height:
      pw_y = height - y - 1;
    else:
      pw_y = tile_size;
              
  fname = '{}/{}_{}_{}_{}_{}.png'.format(output_folder, x, y, pw_x, pw_y, mpp);
  print(fname)
  if os.path.isfile(output_folder+'/'+str(x)+'_'+ str(y)+'_'+ str(pw_x)+'_'+ str(pw_y)+'_'+ str(mpp)+'.png'):
    print(str(x)+'_'+ str(y)+'_'+ str(pw_x)+'_'+ str(pw_y)+'_'+ str(mpp))
    count+=1           
  subprocess.Popen('bash save_tile.sh {} {} {} {} {} {}'.format(slide_name, x, y, pw_x, pw_y, fname), shell=True).wait();        
#################################################   

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
  for x in range(1, width, tile_size):
    for y in range(1, height, tile_size):  
      executor.submit(process_one_tile,x,y);                
  print('finished cropping:', count,count2,count3)  
      
                 
             


