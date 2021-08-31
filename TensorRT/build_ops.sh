#!/bin/bash

# Builds the custom ops in the TensorRT subfolder.
# This is meant to be called from the 3DFeatNet base directory.
set -e

cd TensorRT/grouping
mkdir -p build && cd build
echo "################## Building for Grouping Ops ##################"
cmake ..
make

cd ../../sign
mkdir -p build && cd build
echo "##################   Building for Sign Ops   ##################"
cmake ..
make
