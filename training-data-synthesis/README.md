# Nuclear image synthesis code

This code generates synthetic images with nuclei masks, and real images without masks.  

This code requires a database of relatively large real image tiles under:  
./nuclei_synthesis_40X_online/real_tiles/  

On eagle, this is one under:  
/data08/shared/lehhou/nuclei_synthesis_40X_api_400x400/nuclei_synthesis_40X_online/real_tiles  

## Synthesize fake training images with masks
bash draw_fake_main.sh

## Extract real images with no mask
nohup bash draw_real_main.sh &

