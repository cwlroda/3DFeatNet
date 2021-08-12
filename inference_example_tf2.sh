#!/bin/bash
python inference_tf2.py             \
    --data_dir='./example_data' \
    --output_dir='./example_data/results'   \
    --checkpoint=./ckpt/checkpoint.ckpt \
    --randomize_points