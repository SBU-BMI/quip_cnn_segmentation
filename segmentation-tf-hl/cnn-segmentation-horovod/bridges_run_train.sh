#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-AI
#SBATCH --ntasks-per-node 8
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:volta16:8

#echo commands to stdout
set -x

module load singularity
module load mpi/gcc_openmpi-cuda

CONTAINER=./tf-hvd-bridges.simg


#mpirun -np 2 -hostfile hostfile hostname
#mpirun -np 2 -hostfile hostfile singularity exec $CONTAINER hostname

pwd

echo $LOCAL

pushd $LOCAL
cp /home/lot/scratch/data/1703*.tgz .
tar -xvzf 1703*.tgz
popd

date
#HOROVOD_TIMELINE=horovod_trace.json time mpirun -np 4 -x HOROVOD_TIMELINE -hostfile hostfile --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec $CONTAINER python main.py &> log.train.txt &
~/install/bin/mpirun -np 8 --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec -B /local --nv $CONTAINER python main.py
#mpirun -np 8 --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib singularity exec --nv $CONTAINER hostname
date

exit 0

