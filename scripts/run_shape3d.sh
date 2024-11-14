#!/bin/bash

python train.py \
    --mode seq_train \
    --dataset shapes3d \
    --num_ladders 3 \
    --batch_size 256 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --beta 20 \
    --z_dim 3 \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/shapes3d/ \
    --optim adamw
