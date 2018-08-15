#!/bin/bash

filename="image_list"
IFS=$'\n' read -d '' -r -a file_path < $filename

for image_path in "${file_path[@]}"
do  
  python -u save_tif_to_tiles.py  $image_path './tiles'
done
