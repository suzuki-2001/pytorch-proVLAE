#!/bin/bash

torchrun --nproc_per_node=2 train_ddp.py \
    --distributed True \
    --mode seq_train \
    --dataset ident3d \
    --num_ladders 3 \
    --batch_size 128 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 1 \
    --z_dim 3 \
    --coff 0.5 \
    --hidden_dim 64 \
    --fade_in_duration 5000 \
    --output_dir ./output/ident3d/ \
    --optim adamw
