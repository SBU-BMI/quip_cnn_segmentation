from pymongo import MongoClient
import json 
import csv
import sys
import os    
    
if __name__ == '__main__':
  if len(sys.argv)<1:
    print "usage:python find_segment_count.py image_list"
    exit();    
  
  db_host="seahawk.uhmc.sunysb.edu";  
  db_port="27017";   
  
  client  = MongoClient('mongodb://'+db_host+':'+db_port+'/');      
  
  db = client.quip;  
  objects = db.objects;
  metadata = db.metadata; 
  images = db.images;
  
  image_list_file = sys.argv[-1]; 
  
  print '--- read image_user_list file ---- ';  
  index=0;
  image_list=[];  
  with open(image_list_file, 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    my_list = list(reader);
    for each_row in my_list:       
      #print each_row[0];                     
      image_list.append(each_row[0]);                
  print "total rows from image_list file is %d " % len(image_list) ; 
  #print image_list;
  #exit(); 
  
  for case_id in image_list:
    #print case_id;
    record_count1=metadata.find({"image.case_id":case_id,
                                 "image.subject_id":case_id,
                                 "provenance.analysis_execution_id":"cnn-segmentation-1" }).count();                                                          
    #print case_id,record_count1;  
    record_count2=objects.find({"provenance.image.case_id":case_id,
                                "provenance.image.subject_id":case_id,
                                "provenance.analysis.execution_id":"cnn-segmentation-1"}).count();
    print case_id,record_count1,record_count2;
    '''  
    if record_count1 >0:  
      print "remove objects record for case_id" + str(case_id);                      
      objects.remove({"provenance.image.case_id":case_id,
                      "provenance.image.subject_id":case_id,
                      "provenance.analysis.execution_id":"cnn-segmentation-1"});
                      
      print "remove matadata record for case_id" + str(case_id);                
      metadata.remove({"image.case_id":case_id,
                       "image.subject_id":case_id,
                       "provenance.analysis_execution_id":"cnn-segmentation-1" }) ;               
    
    '''
    
              
 
       
  
  
  
  