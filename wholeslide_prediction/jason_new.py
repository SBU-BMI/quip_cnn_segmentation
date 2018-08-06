import json
import os
import openslide
import collections
import sys
'''
dict_model=json.loads('{ "input_type" : "wsi", "otsu_ratio" : 0.9, "curvature_weight" : 0.8, "min_size" : 3, "max_size" : 200, "ms_kernel" : 20, "declump_type" : 1, "levelset_num_iters" : 100, "mpp" : 0.251, "image_width" : 165335, "image_height" : 81533, "tile_minx" : 63488, "tile_miny" : 14336, "tile_width" : 2048, "tile_height" : 2048, "patch_minx" : 63488, "patch_miny" : 14336, "patch_width" : 2048, "patch_height" : 2048, "output_level" : "mask", "out_file_prefix" : "17032547.17032547.1001272994_mpp_0.251_x63488_y14336", "subject_id" : "17032547", "case_id" : "17032547", "analysis_id" : "wsi:r0.9:w0.8:l3:u200:k20:j1", "analysis_desc" : "" }')
'''

slide=sys.argv[1]
slidesp=slide.split('/')
slide_name=slidesp[len(slidesp)-1].split('.')[0]
print(slide_name)
#slide_name='PC_067_0_1'
#slide='/data01/shared/tcga_analysis/seer_data/images/Hawaii/batch4/'+slide_name+'.svs'
files=os.listdir('./tiles/'+slide_name+'.svs/') #RGB tiles folder
sufix='-algmeta.json'
fsufix='-features.csv'
savefolder='./results/'+slide_name+'/csv/'
if not os.path.exists(savefolder):
  os.system('mkdir '+savefolder)
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
    line='{ "input_type" : "wsi", "otsu_ratio" : 0.9, "curvature_weight" : 0.8, "min_size" : 3, "max_size" : 200, "ms_kernel" : 20, "declump_type" : 1, "levelset_num_iters" : 100, "mpp" : '+str(mpp)+', "image_width" : '+str(width)+', "image_height" : '+str(height)+', "tile_minx" : '+str(x)+', "tile_miny" : '+str(y)+', "tile_width" : '+str(size)+', "tile_height" : '+str(size)+', "patch_minx" : '+str(x)+', "patch_miny" : '+str(y)+', "patch_width" : '+str(size)+', "patch_height" : '+str(size)+', "output_level" : "mask", "out_file_prefix" : "'+name1+'", "subject_id" : "BC_056_0_1", "case_id" : "BC_056_0_1", "analysis_id" : "test1", "analysis_desc" : "" }'
    
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
    dict_model['subject_id']=slide_name
    dict_model['case_id']=slide_name
    dict_model['analysis_id']='test1'
    dict_model['analysis_desc']=''
    
    line=json.dumps(dict_model)
    print(line)
    fid.write(line)
    fid.close()
