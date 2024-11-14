#!/bin/bash

python train.py \
    --mode seq_train \
    --dataset dtd \
    --num_ladders 3 \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 3 \
    --z_dim 3 \
    --coff 0.2 \
    --hidden_dim 64 \
    --fade_in_duration 5000 \
    --output_dir ./output/dtd/ \
    --optim adamw
