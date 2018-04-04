Copy the envireonment:
1. Download the folder ./wholeslide_prediction
2. Copy the follwing files to ./wholeslide_prediction directory and unzip.
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/data.zip 
/nfs/data01/shared/mazhao6/segmentation-tensorflow/wholeslide_pred_version/logs.zip 

How to use?
1.seer_slides.txt has locations of wholeslides that are going to be processed. There are 20 slides. Each of them takes about 8-10 hours for segmenting.
2.1_save_svs_to_tiles.sh save the wholslide image to 4000*4000 tiles, two areguments are needed. The first is the location of one wholeslide image, the second is the folder to save, using './tiles'.
3.2_test_seer_paral_release.sh is to run segmentation on tiles. For segmenting different slides, the following parameters may be changed 
DATASET=BC_069_0_1   #the name of the slide
GPU=1   #which GPU to use



