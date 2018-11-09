#!/bin/bash

CONTAINER=/data1/jlogan/tf/tf-hvd.simg
MPIRUN=/cm/shared/apps/openmpi/gcc/64/1.10.2/bin/mpirun

#mpirun -np 2 -hostfile hostfile hostname
#mpirun -np 2 -hostfile hostfile singularity exec $CONTAINER hostname

CUDA_VISIBLE_DEVICES=1 time $MPIRUN -np 1 -hostfile hostfile --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec $CONTAINER python main.py --is_train=False

exit 0

