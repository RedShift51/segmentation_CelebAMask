#FROM pytorch/pytorch:1.7.1-cuda10.1-cudnn7-devel
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
#FROM ubuntu:16.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
#FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

#RUN add-apt-repository ppa:jonathonf/python-3.6
#RUN apt-get install -y apt-utils

RUN apt-get update -y --fix-missing  && yes | apt-get upgrade -y
RUN apt-get install -y software-properties-common && \
     add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y

RUN mkdir -p /examples
#RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip vim wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py --user
#RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6 --user
RUN rm -f /usr/local/bin/pip3 && rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python3
RUN ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3
RUN pip3 install --upgrade pip

RUN pip3 install tensorflow-gpu==2.3.0
RUN pip3 install numpy scipy matplotlib imageio
RUN pip3 install numpy easydict Cython progressbar2 tensorboardX
RUN pip3 install cython

RUN apt-get install libgl1-mesa-glx -y
RUN pip3 install opencv-python


