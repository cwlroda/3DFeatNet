#/bin/bash

# Call this script from an outside compile wrapper.

NVCC_VER=$1
TF_VER=$2
CXX_ABI_FLAG=$3

/usr/local/cuda-${NVCC_VER}/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
  -I ${TF_INC} \
  -I ${TF_INC}/external/nsync/public \
  -I /usr/local/cuda-${NVCC_VER}/include -lcudart -L /usr/local/cuda-${NVCC_VER}/lib64/ \
  -L${TF_LIB} -l:libtensorflow_framework.so.${TF_VER} -O2 -D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI_FLAG}

# Toggle USE_CXX11_ABI to 0 if there are include errors.
