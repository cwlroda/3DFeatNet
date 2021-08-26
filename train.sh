#/bin/bash

DATASET_DIR=./data/oxford
LOG_DIR=./ckpt_3
GPU_ID=0  # No more multi GPU for now

set -e

# Pretrain
python train.py \
  --data_dir $DATASET_DIR \
  --log_dir $LOG_DIR/pretrain \
  --augmentation Jitter RotateSmall Shift \
  --noattention --noregress \
  --num_epochs 2 \
  --gpu $GPU_ID \
  --validate_every_n_steps 1000

# Second stage training: Performance should saturate in ~60 epochs
python train.py \
  --data_dir $DATASET_DIR \
  --log_dir $LOG_DIR/secondstage \
  --checkpoint $LOG_DIR/pretrain/ckpt \
  --checkpoint_every_n_steps 2000 \
  --restore_exclude detection \
  --augmentation Jitter RotateSmall Shift Rotate1D \
  --num_epochs 70 \
  --gpu $GPU_ID \
  --validate_every_n_steps 250