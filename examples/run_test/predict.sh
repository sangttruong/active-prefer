#!/bin/bash

deepspeed --num_gpus 2 ../../src/train_bash.py \
    --deepspeed ../deepspeed/ds_z3_config.json \
    --stage oracle \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset arc_challenge \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --output_dir ../../saves/LLaMA2-7B/full/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --ddp_timeout 1800000 \
    --plot_loss 
