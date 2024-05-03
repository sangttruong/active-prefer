#!/bin/bash

# Set variables
sanity_check="False"
model_name="meta-llama/Llama-2-7b-hf"
method="random"

# Command to run ARC script
python data/arc/arc.py --sanity_check $sanity_check --model_name $model_name --method $method

python data/truthful_qa/truthful_qa.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"
python data/hellaswag/hellaswag.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"
python data/winogrande/winogrande.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"
python data/mmlu/mmlu.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"
python data/hh_rlhf/hh_rlhf.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"

# python data/reward_bench/reward_bench.py --sanity_check "$sanity_check" --model_name "$model_name" --method "$method"
