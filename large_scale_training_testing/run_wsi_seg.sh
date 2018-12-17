#!/bin/bash

################################
# GPU used for the CNN. If you have 4 GPUs, GPU_ID could be 0, 1, 2, or 3
# Check your GPU availability using nvidia-smi
GPU_ID=0
# Do not change this. It will be looking for ./logs/model_trained/
MODEL=model_trained
# If you run this pipelien on eagle, keep ON_EAGLE=1, Otherwise, keep ON_EAGLE=0
ON_EAGLE=1
# Number of processes for postprocessing (watershed, generating json, csv files etc.)
POSTPROCESS_NPROC=12
# Segmentation threshold (0.0 ~ 1.0). A lower value results in more segmented nuclear material
POSTPROCESS_SEG_THRES=0.33
# Detection threshold (0.0 ~ 1.0). A lower value results in more segmented nuclei
POSTPROCESS_DET_THRES=0.07
# Window size for postprocessing each nuclei
POSTPROCESS_WIN_SIZE=200
# Minimum size of a nucleus. A segmented object smaller than this will be consider as noise and discarded
POSTPROCESS_MIN_NUCLEUS_SIZE=20
# Maximum size of a nucleus. A segmented object bigger than this will be consider as noise and discarded
POSTPROCESS_MAX_NUCLEUS_SIZE=65536
# Description burn into the json meta faile
DESCRIPTION_IN_JSON=seg

# If you already have all segmentation results (*_SEG.png), you might want to do postprocessing only
DO_GPU_PROCESS=True
DO_CPU_POSTPROCESS=True
################################

# The root of all of your data
# WSIs, log files, and outputs will be stored under this folder
LOCAL_DATA_ROOT=/data1/wsi_seg_local_data

INPUT_F=${LOCAL_DATA_ROOT}/svs/
OUTPUT_F=${LOCAL_DATA_ROOT}/seg_tiles/
LOG_F=${LOCAL_DATA_ROOT}/logs/
mkdir -p ${INPUT_F} ${OUTPUT_F} ${LOG_F}

# Prepare input
# If you have images (.svs or .tif) under ${INPUT_F}/ already, you can just leave this line commented out
#scp nfs001:/data/tcga_data/tumor/gbm/TCGA-??-????-???-??-DX8.*.svs ${INPUT_F}/

if [ ${ON_EAGLE} -eq 1 ]; then
    # Call this only if you work on eagle
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
        --do_gpu_process=${DO_GPU_PROCESS} \
        --do_cpu_postprocess=${DO_CPU_POSTPROCESS} \
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
        --do_gpu_process=${DO_GPU_PROCESS} \
        --do_cpu_postprocess=${DO_CPU_POSTPROCESS} \
        &> ${LOG_F}/log_wsi_seg.txt
fi

# Relocate output
# You can relocate the output to a remote host by:
#ssh nfs001 mkdir -p /data/shared/lehhou/wsi_seg_output/
#scp -r ${OUTPUT_F}/* nfs001:/data/shared/lehhou/wsi_seg_output/

exit 0
