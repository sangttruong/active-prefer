CUDA_VISIBLE_DEVICES=1,2 accelerate launch --main_process_port=29506\
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --flash_attn True\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_model_llama_70b \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --eval_steps 10000 \
    --learning_rate 1.5e-4 \
    --num_train_epochs 1\
    --max_samples 20\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16

CUDA_VISIBLE_DEVICES=1,2 accelerate launch --main_process_port=29506\
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
    --output_dir saves/oracle_model_llama_70b \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --eval_steps 10000 \
    --learning_rate 1.5e-4 \
    --num_train_epochs 1\
    --max_samples 20\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16

=============================================================
# 4 card
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-70b-hf  \
    --flash_attn False\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/oracle_model_llama_70b \
    --num_layer_trainable 0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --eval_steps 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 10\
    --ddp_timeout 1800000 \
    --plot_loss \
    --report_to none\
    --fp16