#!/bin/bash

torchrun --nproc_per_node=2 --master_port=29501 src/train.py \
    --distributed \
    --mode seq_train \
    --dataset fashionmnist \
    --optim adamw \
    --num_ladders 3 \
    --batch_size 64 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 3 \
    --z_dim 2 \
    --coff 0.5 \
    --pre_kl \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/fashionmnist/ \
    --data_path ./data/fashionmnist/ 
    