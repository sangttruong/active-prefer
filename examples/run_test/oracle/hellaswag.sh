# hyperturing2

python run_oracle.py --model_name meta-llama/Llama-2-7b-hf --dataset_name winogrande --re_compute_emb True

python run_oracle.py --model_name meta-llama/Meta-Llama-3-8B --dataset_name hellaswag --flash_attn False
CUDA_VISIBLE_DEVICES=8 wandb agent tranquocvinh2611/oracle/g6vpzn1v