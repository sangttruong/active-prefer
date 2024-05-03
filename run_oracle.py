import os

def run_cli_command(command):
    # Run the command
    os.system(command)

def run_oracle():
    # Define 
    num_oracle = 10
    gpu_device = 7
    model_names = [
        "meta-llama/Llama-2-7b-hf"
    ]
    datasets = [
        "arc_challenge"
    ]

    for model_name in model_names:
        for dataset in datasets:
            model_name_short = model_name.split('/')[-1]
            dataset_name = f"{dataset}_train_Llama-2-7b-hf_random"
            output_dir = f"saves/{model_name_short}/{dataset}"


            # Training command
            command = f"""CUDA_VISIBLE_DEVICES={gpu_device} python src/train_bash.py \
                --stage oracle \
                --do_train \
                --flash_attn True \
                --model_name_or_path "{model_name}" \
                --dataset_dir data \
                --dataset "{dataset_name}" \
                --template default \
                --finetuning_type freeze \
                --output_dir "{output_dir}" \
                --overwrite_output_dir \
                --cutoff_len 1024 \
                --per_device_eval_batch_size 2 \
                --is_compute_emb True \
                --num_oracle "{num_oracle}"
            """

            run_cli_command(command)

if __name__ == "__main__":
    run_oracle()
