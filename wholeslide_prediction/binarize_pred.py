import cv2
import os
import sys
import skimage.morphology
import skimage.io
import numpy as np
import glob
import concurrent.futures

max_workers=8;
#slide name
slide=sys.argv[1]
#path for input files: path to CNN prediction masks from step 2 :2_test_seer_paral_release.sh
path1='./segmentation_test_images_'+slide+'/'
#path for output files: path for saving output binarized prediction mask
savepath='./results/'+slide+'/segmask/'
#path for output files: final output csv files and json files saves in csvsavepath
csvsavepath='./results/'+slide+'/csv/'
maskpath=savepath
#path for input files: path to rgb tiles
rgbpath='./tiles/'+slide+'/'
#Path to Yi's code
cpp='/nfs/data01/shared/mazhao6/earth/yi_ori/pathomics_analysis/nucleusSegmentation/app/computeFeaturesCombined'

if not os.path.exists('./results/'):
  os.mkdir('./results/')
if not os.path.exists('./results/'+slide):
  os.mkdir('./results/'+slide)
if not os.path.exists(savepath):
  os.mkdir(savepath)
if not os.path.exists(csvsavepath):
  os.mkdir(csvsavepath)
files=glob.glob(path1+'*png')


#############################################
def process_one_tile(filei,count):
  filei=filei.split('/')
  filei=filei[len(filei)-1]
  if filei[len(filei)-8:len(filei)]=='pred.png':
    count+=1
    print('{}/{}'.format(count,len(files)))
    filei1=filei[0:len(filei)-9]+'.png'
    name=filei1
    name1=name[0:len(name)-3]+'csv'
    if not os.path.isfile(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'):
      rgb=cv2.imread(rgbpath+filei1)#[0:len(filei)-9]+filei1[len(filei)-4:len(filei)]
      print('rgbpath',rgbpath+filei1)
      img=cv2.imread(path1+filei)
      print (path1+filei)
      if img.shape[0]>4000 or img.shape[1]>4000:
        img=cv2.resize(img,(rgb.shape[1],rgb.shape[0]))
      
      ret3,th3 = cv2.threshold(img,96,255,cv2.THRESH_BINARY)#+cv.THRESH_OTSU
      #cv2.imwrite('test1.png',th3)
      th3=np.asarray(th3).astype('uint8')
      ccImage = skimage.morphology.label(th3)
      ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=20)
      ccImage = skimage.morphology.remove_small_holes(ccImage,min_size=2500)
      #skimage.io.imsave('test2.png',ccImage*255)
      
      ccImage=ccImage[:,:,0]
      print(np.max((ccImage*255).astype('uint8')))
      skimage.io.imsave(savepath+filei1,(ccImage*255).astype('uint8'))
      ####csv:
      
      print(name1)
      RGBname=filei1
      colx=name.split('_')[0]
      rowy=name.split('_')[1]
      print(colx)
      print(rowy)
      os.system(cpp+' '+rgbpath+RGBname+' '+maskpath+name+' Y'+' '+name1+' '+str(colx)+' '+str(rowy)+' \n mv '+name1+' '+csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'+'\n rm -r '+name1)
      if not os.path.isfile(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'):
        img1=cv2.imread(savepath+filei1)
        kernel = np.ones((7,7),np.uint8)
        erosion = cv2.erode(img1,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        
        if dilation.shape!=rgb.shape:
          dilation=cv2.resize(dilation,(rgb.shape[1],rgb.shape[0]))
          ret3,dilation = cv2.threshold(dilation,96,255,cv2.THRESH_BINARY)
        print(dilation.shape)
        cv2.imwrite(savepath+filei1,dilation[:,:,0])
        os.system(cpp+' '+rgbpath+RGBname+' '+maskpath+name+' Y'+' '+name1+' '+str(colx)+' '+str(rowy)+' \n mv '+name1+' '+csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'+'\n rm -r '+name1)
        if not os.path.isfile(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'):
          img1=cv2.imread(savepath+filei1)
          kernel = np.ones((20,20),np.uint8)
          erosion = cv2.erode(img1,kernel,iterations = 1)
          dilation = cv2.dilate(erosion,kernel,iterations = 1)
          
          if dilation.shape!=rgb.shape:
            dilation=cv2.resize(dilation,(rgb.shape[1],rgb.shape[0]))
            ret3,dilation = cv2.threshold(dilation,96,255,cv2.THRESH_BINARY)
            
          print(dilation.shape)
          cv2.imwrite(savepath+filei1,dilation[:,:,0])
          os.system(cpp+' '+rgbpath+RGBname+' '+maskpath+name+' Y'+' '+name1+' '+str(colx)+' '+str(rowy)+' \n mv '+name1+' '+csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'+'\n rm -r '+name1)
        
          if not os.path.isfile(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv'):
            print('####################################################################################')
            print('lost tile:'+savepath+filei1)       
      
    else:
      print('skip '+csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv')
#############################################################################################  
    
count=0
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:  
  for filei in files:
    executor.submit(process_one_tile,filei,count);  
 
