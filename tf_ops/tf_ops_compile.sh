#!/bin/bash

# Master script for compilation of custom ops.
# Obtain cuda version (assume that nvcc has been added to path)
set -e
NVCC_VER=`nvcc --version`
NVCC_VER=`echo $NVCC_VER | cut -d "_" -f 7`
NVCC_VER="${NVCC_VER:0:4}"  # Highly hardcoded

# set tensorflow version
TF_VER=`python -c "import tensorflow as tf; print(tf.__version__)"`
TF_VER="${TF_VER:0:1}"

if [ $TF_VER -eq 1 ];
then CXX_ABI_FLAG=1
else CXX_ABI_FLAG=0
fi

# echo ${NVCC_VER}, ${TF_VER}, $CXX_ABI_FLAG
cd grouping
bash tf_grouping_compile.sh ${NVCC_VER} ${TF_VER} $CXX_ABI_FLAG

cd ../sampling
bash tf_sampling_compile.sh ${NVCC_VER} ${TF_VER} $CXX_ABI_FLAG