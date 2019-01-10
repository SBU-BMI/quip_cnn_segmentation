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
  exit();   
  
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
  #image_folder="/data02/shared/cnn_prediction/images/rutgers-2018-11";
  image_folder="nfs002:/data/shared/cnn_prediction/images/rutgers-2018-11";
  result_folder="nfs002:/data/shared/cnn_prediction/results";
  segment_result_folder="/home/bwang/seer_new/wholeslide_prediction/results" 
  destination_folder="/data02/shared/cnn_prediction/results"
  
  #run code for step 0,1,2
  step_num=1
  
  if step_num==0:
    #login in nfs002 and run this portion of code 
    print '--- Step 0: copy image tif file from seahawk to eagle cluster nfs002 node  ---- ';   
    for image_name in image_list:  
      source_path=source_user+"@"+source_server+":"+source_folder + "/" + image_name;
      print source_path
      subprocess.call(['scp', source_path,local_folder]);
  elif step_num==1:   
    #login in master node and run this portion of code
    print '--- Step1 run the CNN pipeline to generate segment result in json/csv file format and then save resutl to nfs002 node ---- '; 
    for image_name in image_list[0:10]: 
      print image_name;   
      cmd = "qsub -v image_id=" + image_name + ",image_path=" + image_folder + " run_cnn.pbs";
      print cmd;
      proc = subprocess.Popen(cmd, shell=True) 
      status = proc.wait() ;   
  elif step_num==2:
    #login in master node and run this portion of code
    print '--- Step 2 LOAD SEGMENT RESULT TO MONGO DATABASE IN SEAHAWK ---- '; 
    for image_name in image_list:   
      cmd = "qsub -v image_id=" + image_name + ",result_path=" + result_folder + " run_load_result.pbs";
      print cmd;
      proc = subprocess.Popen(cmd, shell=True) 
      status = proc.wait() ;         
  exit();
  
  
  
