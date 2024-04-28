# Eval Lora
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train_bash.py \
    --stage rm \
    --do_eval \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --vhead_oracle_path saves/test/multi_oracle/oracle_0 \
    --flash_attn False\
    --dataset reward_bench_test \
    --dataset_dir data \
    --template default \
    --output_dir saves/test/multi_oracle/oracle_0\
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 2 \
    --report_to none\
    --fp16
