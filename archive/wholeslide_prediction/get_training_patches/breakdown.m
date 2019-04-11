%resize images to 0.8, 0.9, 1.0, 1.1, 1.2, crop images to 140x140 with step size 70.
function count=beakdown(im,savepath,count)

if size(size(im),2)<3
  empty=zeros(size(im,1),size(im,2),3);
  empty(:,:,1)=im;
  im=empty;
end
size_list=[0.8,0.9,1.0,1.1,1.2];
for i=1:size(size_list,2)
  im2=imresize(im,size_list(i));
  s1=size(im2,1);
  s2=size(im2,2);
  for j=1:70:s1-140
    for k=1:70:s2-140
      cropped=im2(j:j+140-1,k:k+140-1,:);
      imwrite(cropped,[savepath,num2str(count),'.png']);
      count=count+1
    end
    cropped=im2(j:j+140-1,s2-140+1:s2,:);
    imwrite(cropped,[savepath,num2str(count),'.png']);
    count=count+1;
  end
  
  for k=1:70:s2-140
    cropped=im2(s1-140+1:s1,k:k+140-1,:);
    imwrite(cropped,[savepath,num2str(count),'.png']);
    count=count+1;
  end 
  
  cropped=im2(s1-140+1:s1,s2-140+1:s2,:);
  imwrite(cropped,[savepath,num2str(count),'.png']);
  count=count+1
end
