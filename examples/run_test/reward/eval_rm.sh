------------------------------------------------------
# Eval single gpu
CUDA_VISIBLE_DEVICES=5 python src/train_bash.py \
    --stage rm \
    --do_eval \
    --model_name_or_path saves/oracle_lora_v256/checkpoint-100/ \
    --flash_attn True\
    --dataset reward_bench_test \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_lora_v256\
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
    --learning_rate 5e-6 \
    --num_train_epochs 3\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16


-------------------
# Eval multi gpu
CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage rm \
    --do_eval \
    --model_name_or_path saves/oracle_model_5e-5 \
    --flash_attn False\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_model_5e-5\
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
    --learning_rate 5e-6 \
    --num_train_epochs 3\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16


-------------------
# Eval Lora
CUDA_VISIBLE_DEVICES=8,9 accelerate launch --main_process_port 29510 \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage rm \
    --do_eval \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path saves/oracle_lora_v256_v2 \
    --flash_attn False\
    --dataset reward_bench_test \
    --dataset_dir data \
    --template default \
    --output_dir saves/oracle_lora_v256_v2\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --report_to none\
    --fp16
