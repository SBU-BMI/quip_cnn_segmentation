FROM  tensorflow/tensorflow:1.14.0-gpu
MAINTAINER quip_cnn_segmentation

RUN 	apt-get -y update && \
	apt-get -y install git python-pip openslide-tools wget libsm6 git && \
	pip install openslide-python scikit-image scipy numpy opencv-python==4.2.0.32 tqdm

WORKDIR /root

COPY    . /root/quip_cnn_segmentation/

RUN	cd /root/quip_cnn_segmentation/segmentation-of-nuclei/cnn_model && \
	tar -xzvf model_trained.tar.gz  

ENV	BASE_DIR="/root/quip_cnn_segmentation/"
ENV	PATH="./":$PATH

ENV	MODEL_VER="v1.0"
ENV	MODEL_URL="https://github.com/SBU-BMI/quip_cnn_segmentation/blob/master/segmentation-of-nuclei/cnn_model/model_trained.tar.gz"

WORKDIR /root/quip_cnn_segmentation/segmentation-of-nuclei

CMD ["/bin/bash"]
