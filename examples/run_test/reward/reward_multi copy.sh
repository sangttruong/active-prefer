#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --flash_attn True\
    --dataset truthful_qa_train \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir saves/LLaMA2-7B/test \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_samples 100 \
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16