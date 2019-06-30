#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py &> log.train.txt &

exit 0
