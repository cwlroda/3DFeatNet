#!/bin/bash

# Startup script to run when starting up the TensorRT Docker dev container, to install TensorRT and onnx2trt.

#install TRT
cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)
make install

#install onnx2trt
cd /workspace/onnx-tensorrt
mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=$TRT_OSSPATH && make -j$(nproc)
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
make install

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/tf2/lib/python3.7/site-packages/tensorflow/