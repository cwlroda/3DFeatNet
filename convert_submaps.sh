#!/bin/bash
# Usage: ./convert_submaps.sh <submap_dir>
# e.g. ./convert_submaps.sh coslam/Datasets

dir=$1

if [ ! -d ./local_data ]; then
    mkdir -p ./local_data;
fi;

for d in $dir/*
do
    for dd in $d/*
    do
        python3 submap_converter.py $dd/submap*
    done
done
