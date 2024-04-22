
#### Test
python pipeline.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name reward_bench_train \
    --method random \
    --num_iters 5\
    --num_train_epochs 2\
    --percentage 0.1 \
    --use_accelerate \
    --main_process_port 29505\
    --gpu_ids 6,7\
    --sanity_check\
    --is_compute_emb


#### Full
python pipeline.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name reward_bench_train \
    --method random \
    --num_iters 5\
    --num_train_epochs 2\
    --percentage 0.1 \
    --use_accelerate \
    --main_process_port 29505\
    --gpu_ids 6,7\
    --is_compute_emb
    

CUDA_VISIBLE_DEVICES=3,6 python src/train_bash.py \
    --stage selection\
    --acquisition qbc\
    --do_predict\
    --active_iter 0\
    --model_name_or_path meta-llama/Llama-2-7b-hf\
    --dataset_dir data\
    --dataset reward_bench_train\
    --template default\
    --finetuning_type freeze\
    --output_dir saves/Llama-2-7b-hf/reward_bench_train/random/reward\
    --overwrite_output_dir\
    --cutoff_len 512\
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2\
    --gradient_accumulation_steps 8\
    --lr_scheduler_type cosine\
    --logging_steps 400\
    --warmup_steps 20\
    --save_steps 400\
    --eval_steps 2000\
    --evaluation_strategy steps\
    --learning_rate 5e-05\
    --num_train_epochs 2.0\
    --num_sample_selected 30


-----------------------------------------------
export PATH=/usr/local/cuda-12.2/bin:$PATH

export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export TRITON_CACHE_DIR="/lfs/local/0/sttruong/.triton_1"


export LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/active_rlhf/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/active_rlhf/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/active_rlhf/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/active_rlhf/lib:$LD_LIBRARY_PATH

-----------------------------------------------
