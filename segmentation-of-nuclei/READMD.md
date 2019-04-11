# Large scale training and testing code

## Training
Training input (I put some dummy data in): 

Real patch and masks (400x400 patches):  
./data/nuclei/image_sup  
./data/nuclei/mask_sup 

Synthetic patch, masks, reference real patches, and real patch pairs (400x400 patches extracted by training-data-synthesis):  
./data/nuclei/image  
./data/nuclei/mask  
./data/nuclei/real  
./data/nuclei/refer 

The real and initial synthetic nuclear masks (uint8 png file) for training must have the following binary format:  
  The first bit indicates if it is a nuclear pixel.  
  The second bit indicates if it is a nuclear boundary pixel.  
  The third bit indicates if it is a nuclear center pixel. 
Therefore pixel values in mask files range from 0 to 7. Check out the dummy data in ./data/nuclei/mask/ as examples. 

Once you have training data, just run:  
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py &> log.train.txt & 

## Testing
You need to have a trained model. You can use your trained model under ./logs/, or you can download our trained model:  
wget http://vision.cs.stonybrook.edu/~lehhou/download/model_trained.tar.gz  
Note that the model_trained folder should be under ./logs/, not the content. In other words, the directory should look like:  
ls ./logs/model_trained/  
checkpoint  
graph.pbtxt  
model.ckpt-184780.data-00000-of-00001  
model.ckpt-184780.index  
model.ckpt-184780.meta  
params.json 

Then, just config run_wsi_seg.sh and run:  
nohup bash run_wsi_seg.sh & 

Note: you need to understand run_wsi_seg.sh before running it.  
By default, it assumes that all data is stored under:  
/data1/wsi_seg_local_data/svs/: the folder holds input WSIs (.svs or .tif)  
/data1/wsi_seg_local_data/logs/: the folder holds logs files  
/data1/wsi_seg_local_data/seg_tiles/: the folder holds segmentation outputs 

The python program does everything: tiling, detection and segmentation, csv and json file generation.
You can use visual_seg_polygons.py to visualize polygons overlayed with segmentation results.  
