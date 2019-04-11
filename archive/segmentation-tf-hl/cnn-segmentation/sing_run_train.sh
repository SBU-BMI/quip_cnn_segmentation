#!/bin/bash

GPU=0
CONTAINER=/data1/jlogan/tf/tf-hvd.simg


CUDA_VISIBLE_DEVICES=${GPU} singularity exec $CONTAINER python main.py &> log.train.txt &

exit 0
