FROM ubuntu:18.04

LABEL maintainer "amajidsinar@gmail.com"

WORKDIR /App

RUN apt update &&\
    apt install -y python3 python3-pip wget htop vim &&\
    apt install -y libsm6 libxext6 libxrender-dev git 

RUN pip3 install cython &&\
    pip3 install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html &&\
    pip3 install torchtext==0.7.0