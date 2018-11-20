import json
import os
import openslide
import collections
import sys


slide=sys.argv[1]
slidesp=slide.split('/')
slide_name=slidesp[len(slidesp)-1]
print "slide_name"
print(slide_name)

#remove file extension
if (slide_name.find('.svs') != -1):#svs image file
  image_id=slide_name.replace('.svs','');
elif (slide_name.find('.tif') != -1):#tif image file
 image_id=slide_name.replace('.tif',''); 
else:
  image_id= slide_name        
print "image_id";
print image_id;


files=os.listdir('./tiles/'+slide_name) #RGB tiles folder
sufix='-algmeta.json'
fsufix='-features.csv'
savefolder='./results/'+slide_name+'/csv/'
if not os.path.exists(savefolder):
  os.makedirs(savefolder)
print(slide)
wholeslide=openslide.OpenSlide(slide)
(width,height)=wholeslide.dimensions
count=0

for name in files:
  if name[len(name)-3:len(name)]=='png':
    count+=1
    print('{}/{}'.format(count,len(files)))
    name1=name.split('.png')[0]
    fid=open(savefolder+name1+sufix,'w')
    name=name1.split('_')
    x=int(name[0])
    y=int(name[1])
    size=int(name[2])
    size2=int(name[3])
    mpp=float(name[4])    
    
    dict_model=collections.OrderedDict()
    dict_model['input_type']='wsi'
    dict_model['otsu_ratio']=0.9
    dict_model['curvature_weight']=0.8
    dict_model['min_size']=3
    dict_model['max_size']=200
    dict_model['ms_kernel']=20
    dict_model['declump_type']=1
    dict_model['levelset_num_iters']=100
    dict_model['mpp']=mpp
    dict_model['image_width']=width
    dict_model['image_height']=height
    dict_model['tile_minx']=x
    dict_model['tile_miny']=y
    dict_model['tile_width']=size
    dict_model['tile_height']=size2
    dict_model['patch_minx']=x
    dict_model['patch_miny']=y
    dict_model['patch_width']=size
    dict_model['patch_height']=size2
    dict_model['output_level']='mask'    
    dict_model['out_file_prefix']=name1
    dict_model['subject_id']=image_id
    dict_model['case_id']=image_id
    dict_model['analysis_id']='cnn-segmentation-1'
    dict_model['analysis_desc']=''
    
    line=json.dumps(dict_model)
    print(line)
    fid.write(line)
    fid.close()
