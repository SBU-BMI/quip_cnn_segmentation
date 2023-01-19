FROM  tensorflow/tensorflow:1.14.0-gpu
MAINTAINER quip_cnn_segmentation

RUN 	apt-key del 7fa2af80 && \
	apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && \
	apt-get -y update && \
	apt-get -y install git python-pip openslide-tools wget libsm6 git rsync && \
	pip install openslide-python scikit-image scipy numpy opencv-python==4.2.0.32 tqdm

WORKDIR /quip_app

COPY    . /quip_app/quip_cnn_segmentation/

RUN	cd /quip_app/quip_cnn_segmentation/segmentation-of-nuclei/cnn_model && \
	tar -xzvf model_trained.tar.gz  

ENV	BASE_DIR="/quip_app/quip_cnn_segmentation/"
ENV	PATH="./":$PATH

ENV	MODEL_VER="v1.0"
ENV	MODEL_URL="https://github.com/SBU-BMI/quip_cnn_segmentation/blob/master/segmentation-of-nuclei/cnn_model/model_trained.tar.gz"

WORKDIR /quip_app/quip_cnn_segmentation

CMD ["/bin/bash"]
