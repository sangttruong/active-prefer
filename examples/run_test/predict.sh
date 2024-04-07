#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ../deepspeed/ds_z2_config.json \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset arc_challenge\
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --output_dir ../../saves/LLaMA2-7B/lora/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate \
    --quantization_bit 4

