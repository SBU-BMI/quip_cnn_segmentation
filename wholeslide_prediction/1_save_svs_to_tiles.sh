#!/bin/bash

#echo "Pls enter image_list:"
#read filename

filename="image_list_2018_10_31_new"

IFS=$'\n' read -d '' -r -a file_path < $filename

for image_path in "${file_path[@]}"
do  
  python -u save_tif_to_tiles.py  $image_path './tiles'
done