

python pipeline.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name reward_bench_train \
    --method random \
    --num_iters 2\
    --num_train_epochs 2\
    --percentage 0.1 \
    --gpu_ids 2,3 \
    --use_accelerate \
    --sanity_check \
    --main_process_port 29505
    
