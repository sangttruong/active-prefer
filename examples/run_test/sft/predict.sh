CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test_predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --temperature 1.0\
    --top_p 1\
    --top_k 50\
    --max_samples 4 \
    --predict_with_generate

# ======================================================
CUDA_VISIBLE_DEVICES=3,5 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf\
    --dataset reward_bench_train \
    --dataset_dir data \
    --template default \
    --finetuning_type freeze \
    --output_dir saves/test_predict_2 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --temperature 1.0\
    --top_p 1\
    --top_k 50\
    --max_samples 4 \
    --predict_with_generate