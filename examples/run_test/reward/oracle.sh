CUDA_VISIBLE_DEVICES=6 python src/train_bash.py \
    --stage oracle \
    --do_train \
    --flash_attn False\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test/multi_oracle\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 400 \
    --warmup_steps 20 \
    --save_steps 400 \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 10\
    --is_compute_emb False\
    --num_oracle 10


--------------------------------------------
CUDA_VISIBLE_DEVICES=2,3 python src/train_bash.py \
    --stage oracle \
    --flash_attn False\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test/multi_oracle\\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 400 \
    --warmup_steps 20 \
    --save_steps 400 \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    --learning_rate 5e-2 \
    --num_train_epochs 10\
    --is_compute_emb False\
    --num_oracle 10

--------------------------------------------------------------
CUDA_VISIBLE_DEVICES=8,9 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage oracle \
    --flash_attn False\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test/multi_oracle\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 400 \
    --warmup_steps 20 \
    --save_steps 400 \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 10\
    --is_compute_emb False\
    --num_oracle 10