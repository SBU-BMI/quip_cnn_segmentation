import subprocess
import os
import sys
import csv

if __name__ == '__main__':
  if len(sys.argv)<1:
    print "usage:python run_all_steps.py image_list";
    exit(); 
  
  image_list_file = sys.argv[-1];   
  print '--- read image_list file ---- ';    
  image_list=[];  
  with open(image_list_file, 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    my_list = list(reader);
    for each_row in my_list:                 
      image_list.append(each_row[0]);                
  print "total rows from image_list file is %d " % len(image_list) ; 
  print image_list;
  #exit();   
  
  #before run this code
  #file authorized_keys container ssh public key
  #copy public key from eagle cluster to seahawk server
  #cd seahawk folder: ~/.ssh
  #scp bwang@eagle.bmi.stonybrook.edu:/home/bwang/.ssh/authorized_keys .
  #so python will not prompt password while connecting to remote server 
  
  source_user="bwang";
  source_server="seahawk.uhmc.sunysb.edu";
  source_folder="/home2/ug3_seer/img/rutgers-2018-11"
  local_folder="/data/shared/cnn_prediction/images/rutgers-2018-11" 
  image_folder="/data02/shared/cnn_prediction/images/rutgers-2018-11";
  
  '''
  #login in nfs002 and run this portion of code 
  print '--- Step0: copy image tif file from seahawk to eagle cluster nfs002 node  ---- ';   
  for image_name in image_list:  
    source_path=source_user+"@"+source_server+":"+source_folder + "/" + image_name;
    print source_path
    subprocess.call(['scp', source_path,local_folder]);
  '''  
  
  #login in master node and run this portion of code
  print '--- Step1 ---- '; 
  for image_name in image_list: 
    image_path = os.path.join(image_folder, image_name);
    print image_path;   
    cmd = "qsub -v slide_path=" + image_path + " run_step1.pbs";
    print cmd;
    proc = subprocess.Popen(cmd, shell=True) 
    status = proc.wait() ;
    
  exit();