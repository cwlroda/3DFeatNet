#/bin/bash

# Obtain cuda version (assume that nvcc has been added to path)
set -e
NVCC_VER=`nvcc --version`
NVCC_VER=`echo $NVCC_VER | cut -d "_" -f 7`
NVCC_VER="${NVCC_VER:0:4}"  # Highly hardcoded

/usr/local/cuda-${NVCC_VER}/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
  -I ${TF_INC} \
  -I ${TF_INC}/external/nsync/public \
  -I /usr/local/cuda-${NVCC_VER}/include -lcudart -L /usr/local/cuda-${NVCC_VER}/lib64/ \
  -L${TF_LIB} -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
