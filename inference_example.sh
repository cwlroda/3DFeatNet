#!/bin/bash
python inference.py --data_dir='./example_data' --output_dir='./example_data/results' --checkpoint=./ckpt/secondstage/ckpt/checkpoint.ckpt-210000 --randomize_points
