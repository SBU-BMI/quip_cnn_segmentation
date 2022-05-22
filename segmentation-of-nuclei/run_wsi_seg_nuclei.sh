#!/bin/bash

source ../env.sh

################################
# GPU used for the CNN. If you have 4 GPUs, GPU_ID could be 0, 1, 2, or 3
# Check your GPU availability using nvidia-smi
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    GPU_ID=0
else
    GPU_ID=${CUDA_VISIBLE_DEVICES}
fi

# Do not change this. It will be looking for ./cnn_model/model_trained/
MODEL=model_trained

# Number of processes for postprocessing (watershed, generating json, csv files etc.)
if [[ -z "${NPROCS}" ]]; then
    POSTPROCESS_NPROC=12
else
    POSTPROCESS_NPROC=${NPROCS}
fi

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
# Description burn into the json meta file
DESCRIPTION_IN_JSON=seg

# If you already have all segmentation results (*_SEG.png), you might want to do postprocessing only
DO_GPU_PROCESS=True
DO_CPU_POSTPROCESS=True
################################

# The root of all of your data
# WSIs, log files, and outputs will be stored under this folder
LOCAL_DATA_ROOT=${OUT_DIR}

INPUT_F=${LOCAL_DATA_ROOT}/svs/
OUTPUT_F=${LOCAL_DATA_ROOT}/seg_tiles/
LOG_F=${LOCAL_DATA_ROOT}/logs/
mkdir -p ${INPUT_F} ${OUTPUT_F} ${LOG_F}

# VERSION INFO
export MODEL_PATH="${APP_DIR}/segmentation-of-nuclei/cnn_model/model_trained.tar.gz"
export MODEL_HASH=$(sha256sum $MODEL_PATH | cut -f 1 -d ' ')
export SEG_VERSION=$(git show --oneline -s | cut -f 1 -d ' ')":"$MODEL_VER":"$(sha256sum $MODEL_PATH | cut -c1-7)
export GIT_REMOTE=$(git remote -v | head -n 1 | cut -f 1 -d ' '| cut -f 2)
export GIT_BRANCH=$(git branch | grep "\*" | cut -f 2 -d ' ')
export GIT_COMMIT=$(git show | head -n 1 | cut -f 2 -d ' ')

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

exit 0
