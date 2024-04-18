CUDA_VISIBLE_DEVICES=3,4 python src/train_bash.py \
    --stage rm \
    --do_predict \
    --flash_attn True\
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --vhead_oracle_path saves/Llama-2-7b-hf/truthful_qa_train/random/oracle \
    --finetuning_type full \
    --dataset_dir data \
    --dataset reward_bench_train \
    --template default \
    --output_dir saves/Llama-2-7b-hf/truthful_qa_train/random/oracle  \
    --per_device_eval_batch_size 4 \
    --fp16


