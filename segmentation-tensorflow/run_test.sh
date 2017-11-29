#!/bin/bash

# Set MODEL as one of the folders in logs/
# for example:
# MODEL=generative_2017-11-15_14-29-58
MODEL=
DATASET=miccai16
GPU=0

cp segmentation_test_images/${DATASET}/*[0-9].png segmentation_test_images/
cp segmentation_test_images/${DATASET}/image_resize_list.txt segmentation_test_images/

# On eagle and lired, we installed an updated gclib
#CUDA_VISIBLE_DEVICES=${GPU} \
#LD_LIBRARY_PATH="$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/:$LD_LIBRARY_PATH" \
#        $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so ~/anaconda2/bin/python -u main.py \
#        --is_train=False --load_path=${MODEL} &> log.test.txt
CUDA_VISIBLE_DEVICES=${GPU} ~/anaconda2/bin/python -u main.py \
        --is_train=False --load_path=${MODEL} &> log.test.txt
mkdir -p segmentation_test_images/${DATASET}_${MODEL}/
mv segmentation_test_images/*_pred.png segmentation_test_images/${DATASET}_${MODEL}/

rm -f segmentation_test_images/*.png
rm -f segmentation_test_images/image_resize_list.txt

exit 0
