# Nuclear image synthesis code

This code generates synthetic images with nuclei masks, and real images without masks.  

This code requires a database of relatively large (> 1000x1000 pixels in 40X) real image tiles under:  
./nuclei_synthesis_40X_online/real_tiles/  

## Synthesize fake training images with masks
Generate synthetic images with nuclear masks:  
bash draw_fake.sh  

The synthetic nuclear masks (uint8 png file) have the following binary format:  
```
The first bit indicates if it is a nuclear pixel.  
The second bit indicates if it is a nuclear boundary pixel.  
The third bit indicates if it is a nuclear center pixel. 
```
In other words, a pixel value in nuclear masks ranges from 0 to 7.  

## Extract real images with no mask
Extract real images with no masks:  
python draw_real.py

