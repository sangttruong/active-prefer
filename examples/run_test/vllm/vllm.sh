# vinhtran2611/test_export_model

CUDA_VISIBLE_DEVICES=6 API_PORT=8005 python src/api_demo.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --template default \
    --infer_backend vllm \
    --vllm_enforce_eager



CUDA_VISIBLE_DEVICES=0  python run_vllm.py