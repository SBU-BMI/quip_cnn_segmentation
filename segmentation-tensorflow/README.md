# Nucleus segmentation code in tensorflow  

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

