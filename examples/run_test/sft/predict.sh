CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset reward_bench_train_Llama-2-7b-hf_max_entropy_check \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir saves/LLaMA2-7B/full/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate