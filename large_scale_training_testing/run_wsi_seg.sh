#!/bin/bash

################################
GPU_ID=1
MODEL=model_trained
ON_EAGLE=1
POSTPROCESS_NPROC=12
POSTPROCESS_SEG_THRES=0.33
POSTPROCESS_DET_THRES=0.07
POSTPROCESS_WIN_SIZE=200
POSTPROCESS_MIN_NUCLEUS_SIZE=20
POSTPROCESS_MAX_NUCLEUS_SIZE=65536
DESCRIPTION_IN_JSON=seg
ONLY_POSTPROCESS=False
################################

LOCAL_DATA_ROOT=/data1/wsi_seg_local_data
INPUT_F=${LOCAL_DATA_ROOT}/svs/
OUTPUT_F=${LOCAL_DATA_ROOT}/seg_tiles/
LOG_F=${LOCAL_DATA_ROOT}/logs/
mkdir -p ${INPUT_F} ${OUTPUT_F} ${LOG_F}

# Prepare input
scp nfs001:/data/tcga_data/tumor/gbm/TCGA-??-????-???-??-DX8.*.svs ${INPUT_F}/

# Call this only if you work on eagle
if [ ${ON_EAGLE} -eq 1 ]; then
    module purge
    module load cuda80
    module load openslide/3.4.0
    module load extlibs/1.0.0
    export LIBTIFF_CFLAGS="-I/cm/shared/apps/extlibs/include"
    export LIBTIFF_LIBS="-L/cm/shared/apps/extlibs/lib -ltiff"
    export PATH=/home/lehhou/anaconda2/bin:/home/lehhou/cuda-8.0/bin:${PATH}
    export PYTHONPATH=""

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    LD_LIBRARY_PATH=/home/lehhou/my_libc_env/lib/x86_64-linux-gnu:/home/lehhou/my_libc_env/usr/lib64:/home/lehhou/cuda-8.0/lib64:${LD_LIBRARY_PATH} \
    /home/lehhou/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so \
    /home/lehhou/anaconda2/bin/python -u main.py \
        --is_train=False \
        --seg_path=${INPUT_F} \
        --out_path=${OUTPUT_F} \
        --load_path=${MODEL} \
        --postprocess_nproc=${POSTPROCESS_NPROC} \
        --postprocess_seg_thres=${POSTPROCESS_SEG_THRES} \
        --postprocess_det_thres=${POSTPROCESS_DET_THRES} \
        --postprocess_win_size=${POSTPROCESS_WIN_SIZE} \
        --postprocess_min_nucleus_size=${POSTPROCESS_MIN_NUCLEUS_SIZE} \
        --postprocess_max_nucleus_size=${POSTPROCESS_MAX_NUCLEUS_SIZE} \
        --method_description=${DESCRIPTION_IN_JSON} \
        --only_postprocess=False \
        &> ${LOG_F}/log_wsi_seg.txt
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python -u main.py \
        --is_train=False \
        --seg_path=${INPUT_F} \
        --out_path=${OUTPUT_F} \
        --load_path=${MODEL} \
        --postprocess_nproc=${POSTPROCESS_NPROC} \
        --postprocess_seg_thres=${POSTPROCESS_SEG_THRES} \
        --postprocess_det_thres=${POSTPROCESS_DET_THRES} \
        --postprocess_win_size=${POSTPROCESS_WIN_SIZE} \
        --postprocess_min_nucleus_size=${POSTPROCESS_MIN_NUCLEUS_SIZE} \
        --postprocess_max_nucleus_size=${POSTPROCESS_MAX_NUCLEUS_SIZE} \
        --method_description=${DESCRIPTION_IN_JSON} \
        --only_postprocess=False \
        &> ${LOG_F}/log_wsi_seg.txt
fi

# Relocate output
ssh nfs001 mkdir -p /data/shared/lehhou/wsi_seg_output/
scp -r ${OUTPUT_F}/* nfs001:/data/shared/lehhou/wsi_seg_output/

exit 0
