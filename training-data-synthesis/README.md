# Nuclear image synthesis code

This code generates synthetic images with nuclei masks, and real images without masks.  

This code requires a database of relatively large (> 1000x1000 pixels in 40X) real image tiles under:  
./nuclei_synthesis_40X_online/real_tiles/  

## Synthesize fake training images with masks
Generate synthetic images with nuclear masks:  
bash draw_fake_main.sh  

A pixel value in the nuclear mask can 

## Extract real images with no mask
Extract real images with no masks:  
nohup bash draw_real_main.sh &

