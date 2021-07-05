#FROM pytorch/pytorch:1.7.1-cuda10.1-cudnn7-devel
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
#FROM ubuntu:16.04
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
FROM nvcr.io/nvidia/tensorrt:19.08-py3
#FROM nvidia/tensorrt-labs:frontend
#FROM nvcr.io/nvidia/tensorrt:21.0-py3
#FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

#RUN add-apt-repository ppa:jonathonf/python-3.6
#RUN apt-get install -y apt-utils
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update -y --fix-missing  && yes | apt-get upgrade -y
RUN apt-get install dialog apt-utils -y

RUN apt-get install -y software-properties-common && \
     add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt-get install dialog apt-utils -y
RUN apt-get update -y

RUN mkdir -p /examples
#RUN apt-get install -y apt-utils
#RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip vim wget
RUN apt-get install -y build-essential vim wget python3-venv python3-dev git libssl-dev
# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0/cmake-3.19.0.tar.gz
RUN mv cmake-3.19.0.tar.gz /opt
RUN cd /opt && tar zxvf cmake-3.19.0.tar.gz
RUN cd /opt/cmake-3.19.0 && ./bootstrap && make && make install

#RUN wget https://bootstrap.pypa.io/get-pip.py
#RUN python3.6 get-pip.py --user --no-warn-script-location
#RUN rm -f /usr/local/bin/pip3 && rm -f /usr/bin/python3
#RUN ln -s /usr/bin/python3.6 /usr/bin/python3
#RUN ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3
RUN pip install --upgrade pip

RUN pip install tensorflow-gpu==2.3.0
RUN pip install numpy scipy matplotlib imageio
RUN pip install numpy easydict Cython progressbar2 tensorboardX
RUN pip install cython

RUN apt-get install libgl1-mesa-glx -y
RUN pip install opencv-python keras2onnx onnxruntime
RUN pip install -U tf2onnx
RUN apt-get install -y protobuf-compiler libprotobuf-dev

# onnx2trt
RUN git clone https://github.com/onnx/onnx-tensorrt
#RUN cd onnx-tensorrt/third_party && git clone https://github.com/NVIDIA/TensorRT
RUN cd onnx-tensorrt && git submodule init && git submodule update

RUN wget https://github.com/NVIDIA/TensorRT/archive/refs/tags/v5.1.5.tar.gz
RUN git clone https://github.com/madler/zlib
RUN cd zlib && mkdir build && cd build && cmake -DZLIB_INCLUDE_DIR=~/zlib -DZLIB_LIBRARY=~/zlib/build/zlibstatic.lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release .. && make && make install

RUN tar zxvf v5.1.5.tar.gz && cp -r onnx-tensorrt/third_party/onnx TensorRT-5.1.5/parsers
#RUN ls onnx-tensorrt/third_party/onnx && echo 1
#RUN ls TensorRT-5.1.5/parsers/onnx
#RUN cd TensorRT-5.1.5 && mkdir build && cd build && cmake .. && make && make install
RUN mkdir onnx-tensorrt/build && cd onnx-tensorrt/build && cmake .. -DTENSORRT_ROOT="/opt/tensorrt" && make -j
#RUN mkdir onnx-tensorrt/build && cd onnx-tensorrt/build && cmake .. -DTENSORRT_ROOT="~/onnx-tensorrt/third_party/TensorRT" && make -j
#RUN mkdir onnx-tensorrt/build && cd onnx-tensorrt/build && cmake .. -DTENSORRT_ROOT="/usr/include/x86_64-linux-gnu" && make -j
RUN export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
RUN source ~/.bashrc

#RUN apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.1 \
#    libnvinfer-dev=5.1.5-1+cuda10.1

#RUN ln -s /usr/bin/python /usr/bin/python3
