FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer "amajidsinar@gmail.com"

WORKDIR /App

RUN apt update &&\
    apt install -y python3 python3-pip wget htop vim &&\
    apt install -y libsm6 libxext6 libxrender-dev git 

ADD requirements.txt .

RUN pip3 install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html &&\
    pip3 install -r requirements.txt