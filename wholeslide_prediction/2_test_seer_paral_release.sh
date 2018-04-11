#!/bin/bash

# Set MODEL as one of the folders in logs/
# for example:
# MODEL=generative_2017-11-15_14-29-58

DATASET=BC_378_1_1
GPU=1


INDEX=0
STEP=10000
MODEL=generative_2018-01-02_23-21-39
#BC_056_0_1
mkdir segmentation_test_images_${DATASET}
cp ./tiles/${DATASET}.svs/*.png segmentation_test_images_${DATASET}/
python resize_list.py ${DATASET}
cp ./tiles/${DATASET}.svs/image_resize_list.txt segmentation_test_images_${DATASET}/

#cp ./segmentation_test_images/${DATASET}/*.png segmentation_test_images/
#cp ./segmentation_test_images/${DATASET}/image_resize_list.txt segmentation_test_images/

# On eagle and lired, we installed an updated gclib
CUDA_VISIBLE_DEVICES=${GPU} \
LD_LIBRARY_PATH="/home/lehhou/my_libc_env/lib/x86_64-linux-gnu/:/home/lehhou/my_libc_env/usr/lib64/:$LD_LIBRARY_PATH" \
        /home/lehhou/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so /home/lehhou/anaconda2/bin/python -u main.py \
        --is_train=False --load_path=${MODEL} --mao_step_size=${STEP} --mao_index=${INDEX} --mao_file=${DATASET} --pred_scaling=0.5&> log.wholeslide${DATASET}_${INDEX}.txt
#CUDA_VISIBLE_DEVICES=${GPU} ~/anaconda2/bin/python -u main.py \
        #--is_train=False --load_path=${MODEL} &> log.test.txt

#mkdir -p segmentation_test_images1/${DATASET}_${MODEL}/
#mv segmentation_test_images1/*_pred.png segmentation_test_images1/${DATASET}_${MODEL}/

#rm -f segmentation_test_images1/*.png
#rm -f segmentation_test_images1/image_resize_list.txt

exit 0
