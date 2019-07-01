FROM tensorflow/tensorflow:latest-gpu
MAINTAINER quip_cnn_segmentation

RUN 	apt-get -y update && \
	apt-get -y install git python-pip openslide-tools wget libsm6 && \
	pip install openslide-python scikit-image scipy numpy opencv-python tqdm

WORKDIR /root

COPY    . /root/quip_cnn_segmentation/

RUN	cd /root/quip_cnn_segmentation/segmentation-of-nuclei/cnn_model && \
	wget http://vision.cs.stonybrook.edu/~lehhou/download/model_trained.tar.gz && \ 
	tar -xzvf model_trained.tar.gz && rm -f model_trained.tar.gz 

ENV	BASE_DIR="/root/quip_cnn_segmentation/"
ENV	PATH="./":$PATH
WORKDIR /root/quip_cnn_segmentation/segmentation-of-nuclei

CMD ["/bin/bash"]
