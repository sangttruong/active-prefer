#!/bin/bash
# Lora
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --flash_attn False\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir saves/oracle_model_lora \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 10000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16

---------------------------
# continue training
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --adapter_name_or_path saves/oracle_model_lora \
    --flash_attn True\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir saves/oracle_model_lora_5_epochs \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 10000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 2\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16


--------------------------------
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --flash_attn True\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_model_5e-5 \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 10000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 20\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16


CUDA_VISIBLE_DEVICES=5,6,8,9 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path saves/oracle_model \
    --flash_attn False\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_model_5e-6 \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 10000 \
    --evaluation_strategy steps \
    --learning_rate 5e-3 \
    --num_train_epochs 5\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16

