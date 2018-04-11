import cv2
import os
import sys
import skimage.morphology
import skimage.io
import numpy as np

slide=sys.argv[1]

path1='./segmentation_test_images_'+slide+'/'
#'/nfs/data01/shared/mazhao6/segmentation-tensorflow/segmentation_test_images_miccai16_hem/train16_hem_generative_2017-12-11_15-33-32/'#'/nfs/data01/shared/mazhao6/segmentation-tensorflow/segmentation_test_images_0/pred/'
savepath='./results/'+slide+'/segmask/'#'/nfs/data01/shared/mazhao6/segmentation-tensorflow/segmentation_test_images_miccai16_hem/train16_hem_bi/'#
csvsavepath='./results/'+slide+'/csv1/'
maskpath=savepath
rgbpath='./tiles/'+slide+'.svs/'
cpp='/nfs/data01/shared/mazhao6/earth/yi_ori/pathomics_analysis/nucleusSegmentation/app/computeFeaturesCombined'
if not os.path.exists('./results/'):
  os.mkdir('./results/')
if not os.path.exists('./results/'+slide):
  os.mkdir('./results/'+slide)
if not os.path.exists(savepath):
  os.mkdir(savepath)
if not os.path.exists(csvsavepath):
  os.mkdir(csvsavepath)
files=os.listdir(path1)
count=0
for filei in files:
  if filei[len(filei)-8:len(filei)]=='pred.png':
    count+=1
    print('{}/{}'.format(count,len(files)))
    filei1=filei[0:len(filei)-9]+'.png'
    name=filei1
    name1=name[0:len(name)-3]+'csv'
    if not os.path.isfile(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv') or os.path.getsize(csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv')<3000000:
      
      img=cv2.imread(path1+filei)
      if img.shape[0]>4000:
        img=cv2.resize(img,(4000,4000))
      
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
    else:
      print('skip '+csvsavepath+'/'+name1[0:len(name1)-4]+'-features.csv')
  

  
  
