# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.4-cudnn8-devel-ubuntu18.04

RUN apt-get update;

# Install other stuff
RUN apt-get install sudo;
RUN sudo apt-get -y install tmux wget libssl-dev git libgoogle-glog-dev libjpeg-dev zlib1g-dev


# Install python
RUN apt-get install -y vim python3 python3-pip;

# https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install a newer version of libstdc++
RUN sudo apt-get install -y software-properties-common;
RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test;
RUN sudo apt-get update;
RUN sudo apt-get install -y gcc-9 g++-9;

# Download cmake
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.21.0-rc1/cmake-3.21.0-rc1.tar.gz" -O /opt/cmake-3.21.0-rc1.tar.gz && \
  cd /opt && tar xzf cmake-3.21.0-rc1.tar.gz

# Install cmake
RUN cd /opt/cmake-3.21.0-rc1 && \
  ./bootstrap && \
  make -j 16 && \
  make install

# Install pip librarys
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html;
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html;
RUN pip3 install tensorflow
RUN pip3 install jupyterlab jupyter_http_over_ws
RUN pip3 install pybind11
RUN pip3 install numpy scipy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install fastprogress

#set work dir
WORKDIR /mnt
