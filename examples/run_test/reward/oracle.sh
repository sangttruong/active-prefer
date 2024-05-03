#  meta-llama/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-70b-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10

#  meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-7b-chat-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10


#  meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Meta-Llama-3-8B \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10

#  meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Meta-Llama-3-8B-Instruct \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10

#  meta-llama/Llama-2-13b-hf
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-13b-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb False\
    --num_oracle 10

#  meta-llama/Llama-2-13b-chat-hf
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-13b-chat-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10



# mistralai/Mistral-7B-v0.1
CUDA_VISIBLE_DEVICES=3 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Mistral-7B-v0.1 \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10

# mistralai/Mistral-7B-Instruct-v0.2
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Mistral-7B-Instruct-v0.2 \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10


# google/gemma-7b
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path google/gemma-7b \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/gemma-7b \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10

# google/gemma-7b-it
CUDA_VISIBLE_DEVICES=2 python src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path google/gemma-7b-it \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/gemma-7b-it \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10


# ===========================================
# Deepspeed

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-13b-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --is_compute_emb True\
    --num_oracle 10


# ===========================================
# accelerate

CUDA_VISIBLE_DEVICES=1,3 accelerate launch --main_process_port=29505\
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py\
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-70b-hf \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 400 \
    --warmup_steps 20 \
    --save_steps 400 \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 1\
    --is_compute_emb True\
    --num_oracle 1


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
CUDA_VISIBLE_DEVICES=3,6 accelerate launch \
    --config_file examples/accelerate/default.yaml \
    src/train_bash.py \
    --stage oracle \
    --do_train \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test/Llama-2-7b-hf\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 1 \
    --is_compute_emb True\
    --num_oracle 1


