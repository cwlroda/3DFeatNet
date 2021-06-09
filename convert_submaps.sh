#!/bin/bash

dir=$1

for d in $dir/*
do
    for dd in $d/*
    do
        python3 submap_converter.py $dd/submap*
    done
done