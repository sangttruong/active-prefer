#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch\
    ../../src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir ../../data \
    --dataset arc_sample \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../save/llama2_7b_full_dpo\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type "cosine"  \
    --logging_steps 100 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5  \
    --num_train_epochs 1 \
    --max_samples 2000 \
    --val_size 0.1 \
    --plot_loss \
    --dpo_ftx 1.0\
    --quantization_bit 4

