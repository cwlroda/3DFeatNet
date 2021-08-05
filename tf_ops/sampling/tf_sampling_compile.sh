#/bin/bash

# Call this script from an outside compile wrapper.

NVCC_VER=$1
TF_VER=$2
CXX_ABI_FLAG=$3
TF_INC=$4
TF_LIB=$5
PROTO_INC=$6

printf "\n=== Building Sampling ops using `nvcc --version`"
printf "\n=== And g++ version `g++-7 --version`"

/usr/local/cuda-${NVCC_VER}/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++-7 -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
  -I ${PROTO_INC} \
  -I ${TF_INC} \
  -I ${TF_INC}/external/nsync/public \
  -I /usr/local/cuda-${NVCC_VER}/include -lcudart -L /usr/local/cuda-${NVCC_VER}/lib64/ \
  -L${TF_LIB} -l:libtensorflow_framework.so.${TF_VER} -O2 -D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI_FLAG}

# Toggle USE_CXX11_ABI to 0 if there are include errors.