#!/bin/bash

torchrun --nproc_per_node=2 --master_port=29501 src/train.py \
    --distributed \
    --mode seq_train \
    --dataset mpi3d \
    --optim adamw \
    --num_ladders 3 \
    --batch_size 128 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --beta 8 \
    --z_dim 3 \
    --coff 0.5 \
    --pre_kl \
    --hidden_dim 64 \
    --fade_in_duration 5000 \
    --output_dir ./output/mpi3d/ \
    --data_path ./data/mpi3d/ 
