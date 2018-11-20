#!/bin/bash

echo "Pls enter image_list:"
read filename

IFS=$'\n' read -d '' -r -a file_path < $filename

for image_path in "${file_path[@]}"
do    
  IFS='/' read -a path_array <<< "${image_path}"  
  path_length=${#path_array[@]}  
  
  qsub -v slide_path=$image_path run_step1.pbs
   
done