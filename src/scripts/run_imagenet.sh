#!/bin/bash

torchrun --nproc_per_node=2 --master_port=29501 src/train.py \
    --distributed \
    --mode seq_train \
    --dataset imagenet \
    --optim adamw \
    --num_ladders 4 \
    --batch_size  128 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 1 \
    --z_dim 8 \
    --coff 0.1 \
    --pre_kl \
    --hidden_dim 64 \
    --fade_in_duration 5000 \
    --output_dir ./output/imagenet/ \
    --data_path ./data/imagenet/
