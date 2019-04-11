
-----------------------------------------
Instructions for parallel training on the Eagle cluster
(Please add details if I've left anything out...)
-----------------------------------------

Insure you have access to the singularity container (currently at /data1/jlogan/tf/tf-hvd.simg on each of node001-node010 ), and that the CONTAINER variable in the sing_run_train.sh script points to it.

See below for pointing to training data. This version is configured the same as the original serial version.

Horovod is currently only capable of recognizing gpu0 on each node. Adjust hostfile to reflect nodes on which gpu0 is currently available

Adjust -n argument in mpirun command in sing_run_train.sh to be the number of GPUs you want to use (must have this many available!)

Use the following to start the run (returns immediately):
nohup bash run_train.sh

Use the following to monitor the run:
tail -f log.train.txt



# Nucleus segmentation code in tensorflow  (original README from serial version)

## Training with real data  
The miccai15 training data on eagle is under:  
/data06/shared/lehhou/SU_with_ref_github_data_miccai15/  

The miccai16/17 training data on eagle is under:  
/data06/shared/lehhou/SU_with_ref_github_data_miccai16/  

Put the data you want to use under ./data/nuclei/, then:  
nohup bash run_train.sh &  

## Test a trained model  
A trained model is saved under ./logs/. Specify the model you wanna use in run_test.sh:  
MODEL=generative_2017-11-15_14-29-58  

Specify the dataset you wanna test on in run_test.sh:  
DATASET=miccai16  

Then:  
nohup bash run_test.sh &  

## Postprocessing and evaluate the result  
cd ./segmentation_test_images/eval_code/  

Specify the segmentation result and ground truth you wanna use in postprocessing_and_eval.m:  
ground_truth_folder = '../miccai15/';  
pred_folders = {'../miccai15_generative_2017-11-15_15-05-47'};  

Then:  
nohup bash postprocessing_and_eval.sh &  

