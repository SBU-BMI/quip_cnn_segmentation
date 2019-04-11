%get new training patches
clear all
%path for training images and their masks
path='/nfs/data01/shared/mazhao6/segmentation-tensorflow/xml2mask/xml_to_mask/test_xml_to_mask/';
files=dir([path,'*.jpg']);
savepath='./save/';
savepath_mask='./save_mask/';
if ~exist(savepath)
  mkdir(savepath)
end
if ~exist(savepath_mask)
  mkdir(savepath_mask)
end
count=1
for i=1:size(files,1)
  im=imread([path,files(i).name]);
  mask=imread([path,files(i).name(1:end-4),'_mask.png']);
  count1=breakdown(im,savepath,count);
  count1=breakdown(mask,savepath_mask,count);
  count=count1;
end

  
  
  
