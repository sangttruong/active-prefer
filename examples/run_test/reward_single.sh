#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python ../../src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset arc_challenge \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir ../../saves/LLaMA2-7B/lora/reward \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --max_samples 2000 \
    --val_size 0.1 \
    --plot_loss \
    --quantization_bit 4\
    --only_training_vhead True
