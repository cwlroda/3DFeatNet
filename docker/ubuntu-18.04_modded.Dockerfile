# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified by Tianyi to allow for a more suitable development environment for TensorRT optimization

ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=18.04

# Disable warning about "apt-utils" not being installed
ARG DEBIAN_FRONTEND=noninteractive

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.0.1.6
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Allow apt to write to /tmp
RUN mkdir -p /tmp && chmod 1777 /tmp

# Install requried libraries
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update && apt install -y --no-install-recommends \
    libcurl4-openssl-dev wget zlib1g-dev git pkg-config sudo ssh libssl-dev \
    pbzip2 pv bzip2 unzip devscripts lintian fakeroot dh-make build-essential \
    nano tree libprotobuf-dev protobuf-compiler 

# installs miniconda
RUN apt update --fix-missing && apt install -y ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /.bashrc && \
    echo "conda activate base" >> /.bashrc

RUN apt install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt clean

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# update conda
RUN conda update -n base -c defaults conda

# end install miniconda

# Install TensorRT
RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub &&\
    apt update &&\
    sudo apt install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v}

RUN conda init bash
RUN conda create --name tf2 python=3.7

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tf2", "/bin/bash", "-c"]

# Install PyPI packages from requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
# Workaround to remove numpy installed with tensorflow
# RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

# Add path for trtexec to image
ENV PATH="/usr/lib/bin/:$PATH"
# for tensorflow to include the right path
ENV LD_LIBRARY_PATH="/opt/conda/envs/tf2/lib/python3.7/site-packages/tensorflow/:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# Enable the conda env as well as colour terminal
RUN echo "Recovering things at the end"
RUN sed -i "s|activate base|activate tf2|g" /.bashrc
RUN sed -i "s|#force_color|force_color|g" /.bashrc

USER trtuser
RUN ["/bin/bash"]