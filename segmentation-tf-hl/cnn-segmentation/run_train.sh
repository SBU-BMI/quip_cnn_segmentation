#!/bin/bash

GPU=0

source activate tensorflow-1.4

# On eagle and lired, we installed an updated gclib
CUDA_VISIBLE_DEVICES=${GPU} \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/my_libc_env/lib/x86_64-linux-gnu/:${HOME}/my_libc_env/usr/lib64/" \
    ${HOME}/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so \
    ~/anaconda2/bin/python main.py &> log.train.txt
#CUDA_VISIBLE_DEVICES=${GPU} ~/anaconda2/bin/python main.py &> log.train.txt

exit 0
