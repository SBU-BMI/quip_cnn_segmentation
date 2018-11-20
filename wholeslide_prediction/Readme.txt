Copy the envireonment:
1. Download the folder ./wholeslide_prediction
2. Copy the follwing files in the eagle cluster to ./wholeslide_prediction directory and unzip.
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/data.zip 
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/logs.zip 

How to use?
1. Create tiles and results folder in code base folder;
2. Create image list file with image name only
3. Change variable name such as folder location in file run_all_steps.py
4. Change step_num in file run_all_steps.py
5. Run the code with cmd "python run_all_steps.py image_list"
6. Repeat step 4 and 5;
