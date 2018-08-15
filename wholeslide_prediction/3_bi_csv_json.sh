#!/bin/bash

echo "Pls enter image_list:"
read filename

IFS=$'\n' read -d '' -r -a file_path < $filename

for image_path in "${file_path[@]}"
do    
  IFS='/' read -a path_array <<< "${image_path}"  
  path_length=${#path_array[@]}
  case_id=${path_array[$path_length-1]}
  SLIDENAME=$case_id   
  SLIDEPATH=$image_path   
  nohup python -u binarize_pred_old.py ${SLIDENAME} >log.bi.${SLIDENAME}.txt &
  nohup python jason_new.py ${SLIDEPATH} &  
done
