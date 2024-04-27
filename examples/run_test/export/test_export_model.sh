CUDA_VISIBLE_DEVICES=5 python src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path saves/Llama-2-7b-hf/reward_bench_train/random/dpo \
    --template default \
    --finetuning_type lora \
    --export_dir saves/Llama-2-7b-hf/reward_bench_train/random/dpo/full \
    --export_size 2 \
    --export_legacy_format False\
    --export_hub_model_id vinhtran2611/test_export_model
