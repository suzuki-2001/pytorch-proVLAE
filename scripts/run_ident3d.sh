#!/bin/bash

python train.py \
    --mode seq_train \
    --dataset ident3d \
    --num_ladders 3 \
    --batch_size 128 \
    --num_epochs 15 \
    --learning_rate 5e-4 \
    --beta 3 \
    --z_dim 3 \
    --coff 0.3 \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/ident3d/ \
    --optim adamw \
