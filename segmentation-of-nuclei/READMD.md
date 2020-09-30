# docker container for training and prediction 

## Extracting segmentation mask from output folder

Segmentation results are stored in polygon format. To extract segmentation as a numpy array mask, please use extract_patch_segmentation_mask.py:

```python
from extract_patch_segmentation_mask import extract_segmentation_mask

# Provide the output folder path, <x, y>, and patch_width of the location you want to extract mask from.
mask = extract_segmentation_mask(segmentation_polygon_folder, x, y, patch_width)
```

We have released segmentation results (a bunch of segmentation_polygon_folder) for over 6,000 WSIs in 14 cancer types under [this box folder](https://stonybrookmedicine.box.com/s/7n9gdy3i6qmm638or7lbxrzzydb1iv9b). Note that segmentation in COAD, READ, STAD, and UVM are suboptimal.

## Prediction (Segmentation)

Run the container as:

```sh
nvidia-docker run --name quip-segmentation -itd \
  -v /<host-data-folder>:/data/wsi_seg_local_data \
  -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
  -e NPROCS=${num_cpu_cores_for_watershed_processing} \
  quip_cnn_segmentation run_wsi_seg.sh
```

You `host-folder` should have an `svs` subfolder. Input images should be in the svs subfolder. 

Argument `-v /<host-data-folder>:/data/wsi_seg_local_data` maps the `host-folder` on the host to the 
local folder in the container. The segmentation run will create two output folders: 

```
/<host-data-folder>/logs (/data/wsi_seg_local_data/logs): log files are stored in this folder.
/<host-data-folder>/seg_tiles (/data/wsi_seg_local_data/seg_tiles): segmentation output is stored 
in this folder
```

The docker container does everything: tiling, detection and segmentation, csv and json file generation.

## Training
Training input (I put some dummy data in): 

Real patch and masks (400x400 patches):  
```
./data/nuclei/image_sup  
./data/nuclei/mask_sup 
```

Synthetic patch, masks, reference real patches, and real patch pairs (400x400 patches extracted by training-data-synthesis):  
```
./data/nuclei/image  
./data/nuclei/mask  
./data/nuclei/real  
./data/nuclei/refer 
```

The real and initial synthetic nuclear masks (uint8 png file) for training must have the following binary format:  
```
  The first bit indicates if it is a nuclear pixel.  
  The third bit indicates if it is a nuclear center pixel. 
```
Therefore pixel values in mask files range from 0 to 5. Check out the dummy data in ./data/nuclei/mask/ as examples. 

Once you have training data, just run:  
```sh
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py &> log.train.txt & 
```
