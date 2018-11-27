#!/bin/bash

fold=${1}
ctype=${2}

mkdir -p ${fold}/output
mkdir -p ${fold}/nuclei_synthesis_40X_online
cp *.sh *.m *.py ${fold}
cp nuclei_synthesis_40X_online/*.py ${fold}/nuclei_synthesis_40X_online

cd ${fold}
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/detect
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/image
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/mask
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/refer
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/output/real
cd output
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/detect detect
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/image image
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/mask mask
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/refer refer
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/output/real real
cd ../nuclei_synthesis_40X_online/
mkdir -p /dev/shm/lehhou/nuclei_synthesis_40X_api/
rm -rf /dev/shm/lehhou/nuclei_synthesis_40X_api/real_tiles_${ctype}
scp -r nfs008:/data/shared/lehhou/nuclei_synthesis_40X_api/nuclei_synthesis_40X_online/real_tiles_cp_${ctype} /dev/shm/lehhou/nuclei_synthesis_40X_api/real_tiles_${ctype}
ln -s /dev/shm/lehhou/nuclei_synthesis_40X_api/real_tiles_${ctype} real_tiles

exit 0
