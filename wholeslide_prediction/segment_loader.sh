#!/bin/bash


echo "Pls enter image_list:"
read filename

IFS=$'\n' read -d '' -r -a file_path < $filename

source_datafile_path="/home/bwang/seer_new/wholeslide_prediction/results"

#load all data files to mongo DB
for image_path in "${file_path[@]}"
do  
  IFS='/' read -a path_array <<< "${image_path}"  
  path_length=${#path_array[@]}
  case_id=${path_array[$path_length-1]}
  echo $case_id;   
  new_path=$source_datafile_path/$case_id/csv
  echo "$new_path";
  echo "------------------------------------------------------------------------------"  
  /home/tkurc/work/pathomics_featuredb/build/install/featuredb-loader/bin/featuredb-loader \
    --inptype csv \
    --dbhost seahawk.uhmc.sunysb.edu \
    --dbname quip \
    --fromdb \
    --quip $new_path/        
done

exit 0