#!/bin/bash 

source ./env.sh

if [ -d "${APP_DIR}" ]
then
	rm -fr ${APP_DIR}
fi
mkdir -p ${APP_DIR}

rsync -avq /quip_app/quip_cnn_segmentation/ ${APP_DIR}/

cd ${APP_DIR}/segmentation-of-nuclei
./run_wsi_seg_nuclei.sh

cd ${OUT_DIR} 
rm -fr ${APP_DIR}
