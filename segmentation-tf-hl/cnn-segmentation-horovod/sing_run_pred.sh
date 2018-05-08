#!/bin/bash

CONTAINER=/data1/jlogan/tf/tf-hvd.simg


#mpirun -np 2 -hostfile hostfile hostname
#mpirun -np 2 -hostfile hostfile singularity exec $CONTAINER hostname

mpirun -np 1 -hostfile hostfile --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec $CONTAINER python predict.py 

exit 0

