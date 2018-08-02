#!/bin/bash

CONTAINER=/data1/jlogan/tf/tf-hvd.simg


#mpirun -np 2 -hostfile hostfile hostname
#mpirun -np 2 -hostfile hostfile singularity exec $CONTAINER hostname

HOROVOD_TIMELINE=horovod_trace.json time mpirun -np 4 -x HOROVOD_TIMELINE -hostfile hostfile --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec $CONTAINER python main.py &> log.train.txt &

exit 0

