#FROM pytorch/pytorch:1.7.1-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /examples

RUN apt-get install -y git python3-pip vim
RUN pip3 install --upgrade pip

RUN pip install tensorflow-gpu==2.3.0
RUN pip install numpy scipy matplotlib imageio
RUN pip install numpy easydict Cython progressbar2 tensorboardX
RUN pip install cython

RUN apt-get install libgl1-mesa-glx -y
RUN pip install opencv-python


