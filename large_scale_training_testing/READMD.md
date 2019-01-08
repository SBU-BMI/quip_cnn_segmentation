# Large scale training and testing code

## Training
This training code cannot be run on eagle, due to large GPU memory requirements.  
Training input (I put some dummy data in): 

Real patch and masks (400x400 patches extracted by real-training-data-extraction/):  
./data/nuclei/image_sup  
./data/nuclei/mask_sup 

Synthetic patch, masks, reference real patches, and real patch pairs (400x400 patches extracted by training-data-synthesis):  
./data/nuclei/image  
./data/nuclei/mask  
./data/nuclei/real  
./data/nuclei/refer 


## Testing
You need to have a trained model. One is available on eagle. In the code environment:  
cp -r /data01/shared/lehhou/large_scale_training_testing_trained_model/model_trained ./logs/  
You can also download the trained model online.  
Note that the model_trained folder should be under ./logs/, not the content. In other words, the directory should look like:  
ls ./logs/model_trained/model.ckpt-184780.index  
checkpoint  graph.pbtxt  model.ckpt-184780.data-00000-of-00001  model.ckpt-184780.index  model.ckpt-184780.meta  params.json 

Then, just config run_wsi_seg.sh and run:  
nohup bash run_wsi_seg.sh & 

Note: you need to understand run_wsi_seg.sh before running it.  
By default, it assumes that all data is stored under:  
/data1/wsi_seg_local_data/svs/: the folder holds input WSIs (.svs or .tif)  
/data1/wsi_seg_local_data/logs/: the folder holds logs files  
/data1/wsi_seg_local_data/seg_tiles/: the folder holds segmentation outputs 

The python program does everything: tiling, detection and segmentation, csv and json file generation.
You can use visual_seg_polygons.py to visualize polygons overlayed with segmentation results.  
