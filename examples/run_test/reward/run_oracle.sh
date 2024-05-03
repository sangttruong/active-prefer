#!/bin/bash

# Define variables
model_name="meta-llama/Llama-2-7b-hf"
dataset="arc_challenge"

model_name_short="Llama-2-7b-hf_random"
dataset_name="${dataset}_train_${model_name_short}"
output_dir="saves/${model_name}/${dataset_name}"
num_oracle=10
gpu_device=7

# Set CUDA visible devices locally
CUDA_VISIBLE_DEVICES="$gpu_device"

# Training command
python src/train_bash.py \
    --stage oracle \
    --do_train \
    --flash_attn True \
    --model_name_or_path "$model_name" \
    --dataset_dir data \
    --dataset "$dataset_name" \
    --template default \
    --finetuning_type freeze \
    --output_dir "$output_dir" \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True \
    --num_oracle "$num_oracle"
