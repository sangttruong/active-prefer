#!/bin/bash


CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --flash_attn False\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves\
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_samples 100 \
    --val_size 0.1 \
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16