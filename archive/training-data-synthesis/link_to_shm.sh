#!/bin/bash

mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/image
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/refer
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/mask
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/detect

ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/image output/image
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/refer output/refer
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/mask  output/mask
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/detect output/detect

cp -r nuclei_synthesis_40X_online/real_tiles_cp /dev/shm/lehhou/nuclei_synthesis_40X_api/real_tiles
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/real_tiles nuclei_synthesis_40X_online/real_tiles

exit 0
