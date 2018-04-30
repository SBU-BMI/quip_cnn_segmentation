Copy the envireonment:
1. Download the folder ./wholeslide_prediction
2. Copy the follwing files in the eagle cluster to ./wholeslide_prediction directory and unzip.
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/data.zip 
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/logs.zip 

How to use?
1.seer_slides.txt has locations of wholeslides that are going to be processed. There are 20 slides. Each of them takes about 8-10 hours for segmenting.
2.1_save_svs_to_tiles.sh save the wholslide image to 4000*4000 tiles, two areguments are needed. The first is the location of one wholeslide image, the second is the folder to save, using './tiles'.
Using 'nohup bash 1_save_svs_to_tiles.sh &' to run it.
3.2_test_seer_paral_release.sh is to run segmentation on tiles. 
For segmenting different slides, the following parameters should be changed for each slide. 
DATASET=BC_065_0_1  #the name of the slide without '.svs'
GPU=1 #which GPU to use 
Using 'nohup bash 2_test_seer_paral_release.sh &' to run it.
4.3_bi_csv_json.sh is to binarize the CNN prediction, generate polygon and json files.
SLIDENAME and SLIDEPATH should be changed for each slide.
5.In binarize_pred.py line 16, cpp='/nfs/data01/shared/mazhao6/earth/yi_ori/pathomics_analysis/nucleusSegmentation/app/computeFeaturesCombined' is the path for Yi's code, you should change this path according to your environment.
6. The final results for each slide are in ./results/{SLIDENAME}/csv. For each RGB tile, there should be a .json file and a .csv file in this folder. Please make sure the number of these files are correct. Then copy these folders to eagle cluster. For each slide, the directory of this folder on eagle and the directory of the original slide is needed for showing them on the caMicroscope.



