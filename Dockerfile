
FROM openvino/ubuntu20_dev:latest
ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update -y
RUN apt install -y  libgl1 libglib2.0-0  libx11-dev python3-tk python3-dev python3-opencv 
RUN apt install wget
RUN pip3 install kaggle 
RUN pip3 install torch torchvision
RUN pip3 install opencv-python pycocotools  
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git 

WORKDIR /workspace 
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 

RUN mkdir /root/.kaggle/
COPY kaggle.json  /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

RUN kaggle datasets download rtatman/stamp-verification-staver-dataset
RUN apt install unzip
RUN unzip stamp-verification-staver-dataset.zip
RUN rm -f stamp-verification-staver-dataset.zip

COPY train.py /workspace


