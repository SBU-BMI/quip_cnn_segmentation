import sys
import os
index=sys.argv[1]
files=os.listdir('./'+index)
list_txt=open('./'+index+'/image_resize_list.txt','w')
for i in range(len(files)):
  if files[i][len(files[i])-3:len(files[i])]=='png':
    list_txt.write(files[i]+' 1\n')
list_txt.close()
  