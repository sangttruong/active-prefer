CUDA_VISIBLE_DEVICES=6 python src/train_bash.py \
    --stage selection \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-7b-hf\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --active_iter 0\
    --acquisition max_entropy

# ================================
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage selection \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --flash_attn\
    --dataset_dir data \
    --dataset  reward_bench_train\
    --template default \
    --finetuning_type freeze \
    --output_dir saves/Llama-2-7b-hf\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --active_iter 2\
    --acquisition max_entropy