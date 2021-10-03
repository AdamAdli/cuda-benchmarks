# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
# FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update;

# Install other stuff
RUN apt-get install sudo;

# Install python
RUN apt-get install -y vim python3 python3-pip;
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html;
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html;
RUN pip3 install jupyterlab

# install a newer version of libstdc++
RUN sudo apt-get install -y software-properties-common;
RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test;
RUN sudo apt-get update;
RUN sudo apt-get install -y gcc-9 g++-9;


# Add user
ARG UNAME=ybgao
ARG UID=6073
ARG GID=31075
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN echo "ybgao:password" | chpasswd

RUN adduser $UNAME sudo

USER $UNAME

CMD /bin/bash
