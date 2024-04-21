# vinhtran2611/test_export_model

CUDA_VISIBLE_DEVICES=3,4 API_PORT=8005 python src/api_demo.py \
    --model_name_or_path saves/Llama-2-7b-hf/reward_bench_train/random/dpo/full \
    --template llama2 \
    --infer_backend vllm \
    --vllm_enforce_eager


python src/api_demo.py \
    --model_name_or_path saves/Llama-2-7b-hf/reward_bench_train/random/dpo/full \
    --template llama2 \
    --infer_backend vllm \
    --vllm_enforce_eager \
    --gpu_ids 3,4\
    --api_port 8005\