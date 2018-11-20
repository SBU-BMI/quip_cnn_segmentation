#!/bin/bash


echo "Pls enter image_list:"
read filename

IFS=$'\n' read -d '' -r -a file_path < $filename

for image_path in "${file_path[@]}"
do    
  IFS='/' read -a path_array <<< "${image_path}"  
  path_length=${#path_array[@]}
  case_id=${path_array[$path_length-1]}    
  echo $case_id  
  echo $image_path
  qsub -v caseid=$case_id,slide_path=$image_path run_step3.pbs   
done