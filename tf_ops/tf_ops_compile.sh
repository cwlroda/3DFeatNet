#!/bin/bash

# Master script for compilation of custom ops.
# Obtain cuda version (assume that nvcc has been added to path)
set -e

GCC_VER=`g++-7 --version | cut -d " " -f 4`
GCC_VER=${GCC_VER:0:5}
GCC_VERNUM=${GCC_VER:0:1}
if [ $GCC_VERNUM -eq 7 ];
then echo "";
else echo "g++ Version is $GCC_VER. Please use g++ 7 to avoid grief"
    exit 1
fi

NVCC_VER=`nvcc --version`
NVCC_VER=`echo $NVCC_VER | cut -d "_" -f 7`
NVCC_VER="${NVCC_VER:0:4}"  # Highly hardcoded
# Duct tape for the system:
# NVCC_VER="11.4"

# set tensorflow version
OUTPUT=($(python3 -c "import tensorflow as tf; out_list=[tf.sysconfig.get_include(), tf.sysconfig.get_lib(), tf.__version__]; print(out_list)" | tr -d '[],'))
TF_INC=${OUTPUT[0]:1:-1}
TF_LIB=${OUTPUT[1]:1:-1}
TF_VER=${OUTPUT[2]:1:-1}
TF_VERNUM=${TF_VER:0:1}

PROTO_INC=$(which python)
PROTO_INC=$(python3 -c "dir='${PROTO_INC}'; dir_list=dir.split('/'); dir_list.pop(); dir_list.pop(), dir_list.append('include'); print('/'.join(dir_list))")

if [ $TF_VERNUM -eq 1 ];
then CXX_ABI_FLAG=1
else CXX_ABI_FLAG=0
fi

CXX_ABI_FLAG=0

echo "NVCC version: ${NVCC_VER}, Tensorflow version: ${TF_VER}, CXX ABI Flag: ${CXX_ABI_FLAG}"

python -c "exit(1) if (input('>>> Press Enter to proceed, anything else to quit <<<')!='') else exit(0)"

cd grouping
bash tf_grouping_compile.sh ${NVCC_VER} ${TF_VERNUM} ${CXX_ABI_FLAG} ${TF_INC} ${TF_LIB} ${PROTO_INC}

cd ../sampling
bash tf_sampling_compile.sh ${NVCC_VER} ${TF_VERNUM} ${CXX_ABI_FLAG} ${TF_INC} ${TF_LIB} ${PROTO_INC}
