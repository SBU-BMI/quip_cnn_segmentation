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
You need to have a trained model. One is available on eagle:  
cp -r /data01/shared/lehhou/large_scale_training_testing_trained_model/model_trained ./logs/  
You can also download the trained model online.  
Note that the model_trained folder should be under ./logs/, not the content. In other words, the directory should look like:  
ls ./logs/model_trained/  
checkpoint  graph.pbtxt  model.ckpt-184780.data-00000-of-00001  model.ckpt-184780.index  model.ckpt-184780.meta  params.json 

Then, just config run_wsi_seg.sh and run:  
nohup bash run_wsi_seg.sh & 

Note: you need to understand run_wsi_seg.sh before running it.  
It first copies svs files from a remote host, then runs CNN to generate csv and json files,  
then it copies the resulting files to a remote host. 

The python program does everything: tiling, detection and segmentation, csv and json file generation.
You can use visual_seg_polygons.py to visualize polygons overlayed with segmentation results.  
