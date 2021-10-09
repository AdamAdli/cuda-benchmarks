# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
# FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update;

# Install other stuff
RUN apt-get install sudo;
RUN sudo apt-get -y install tmux wget libssl-dev git libgoogle-glog-dev


# Install python
RUN apt-get install -y vim python3 python3-pip;
# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html;
# RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html;
RUN pip3 install jupyterlab
RUN pip3 install pybind11

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


# install sputnik
RUN mkdir /thirdparty
WORKDIR thirdparty
RUN git clone --recursive https://github.com/google-research/sputnik.git && \
	mkdir sputnik/build
WORKDIR /thirdparty/sputnik/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF \
	-DCUDA_ARCHS="60;70" -DCMAKE_INSTALL_PREFIX=/usr/local/sputnik && \
	make -j16 install

#set up env
ENV PYTHONPATH="/mount:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/local/sputnik/lib:${LD_LIBRARY_PATH}"

# Add user
ARG UNAME=tian
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN echo "tian:password" | chpasswd

RUN adduser $UNAME sudo

USER $UNAME

CMD /bin/bash

#set work dir
WORKDIR /mnt
