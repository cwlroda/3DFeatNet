#!/bin/bash

# convenience script to (re)build the custom ops

cd ./tf_ops/grouping
bash tf_grouping_compile.sh

cd ../sampling
bash tf_sampling_compile.sh