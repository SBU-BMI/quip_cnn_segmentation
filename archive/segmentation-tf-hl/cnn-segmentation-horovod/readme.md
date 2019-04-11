-----------------------------------------
Instructions for parallel training on the Eagle cluster
(Please add details if I've left anything out...)
-----------------------------------------

Insure you have access to the singularity container (currently at /data1/jlogan/tf/tf-hvd.simg on each of node001-node010 ), and that the CONTAINER variable in the sing_run_train.sh script points to it.
 
Modify config.py to point to your training data. In particular, train_data_dir and train_mask_dir should point to directories containing images and masks, and the two directories should contain identically named sets of files.

Horovod is currently only capable of recognizing gpu0 on each node. Adjust hostfile to reflect nodes on which gpu0 is currently available

Adjust -n argument in mpirun command in sing_run_train.sh to be the number of GPUs you want to use (must have this many available!)

Use the following to start the run (returns immediately):
nohup bash sing_run_train.sh

Use the following to monitor the run:
tail -f log.train.txt


