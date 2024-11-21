#!/bin/bash

torchrun --nproc_per_node=2 --master_port=29501 src/train.py \
    --distributed \
    --mode indep_train \
    --train_seq 3 \
    --dataset shapes3d \
    --optim adamw \
    --num_ladders 3 \
    --batch_size 128 \
    --num_epochs 16 \
    --learning_rate 5e-4 \
    --beta 1 \
    --z_dim 3 \
    --coff 0.5 \
    --pre_kl \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/shapes3d/ \
    --data_path ./data/ \
    --use_wandb \
    --wandb_project PRO-VLAE \
    --use_kl_annealing \
    --kl_annealing_mode sigmoid \
    --cycle_period 4 \
    --ratio 0.5 \
    --max_kl_weight 1.0 \
    --min_kl_weight 0.1 \
    --num_workers 16 
    