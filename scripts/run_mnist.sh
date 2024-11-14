#!/bin/bash

python train.py \
    --mode seq_train \
    --dataset mnist \
    --num_ladders 3 \
    --batch_size 16 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 3 \
    --z_dim 3 \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/mnist/ \
    --optim adamw