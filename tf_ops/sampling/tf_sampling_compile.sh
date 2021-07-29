#/bin/bash

# Obtain cuda version (assume that nvcc has been added to path)
set -e
NVCC_VER=`nvcc --version`
NVCC_VER=`echo $NVCC_VER | cut -d "_" -f 7`
NVCC_VER="${NVCC_VER:0:4}"  # Highly hardcoded

# set tensorflow version
TF_VER=`python -c "import tensorflow as tf; print(tf.__version__)"`
TF_VER="${TF_VER:0:1}"

/usr/local/cuda-${NVCC_VER}/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
  -I ${TF_INC} \
  -I ${TF_INC}/external/nsync/public \
  -I /usr/local/cuda-${NVCC_VER}/include -lcudart -L /usr/local/cuda-${NVCC_VER}/lib64/ \
  -L${TF_LIB} -l:libtensorflow_framework.so.${TF_VER} -O2 -D_GLIBCXX_USE_CXX11_ABI=1
