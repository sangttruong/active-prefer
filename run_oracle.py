import os
import argparse
import wandb

def run_cli_command(command):
    # Run the command
    print(command)
    os.system(command)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--dataset_name", type=str, default="reward_bench", help="Test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Test")
    parser.add_argument("--flash_attn", type=str, default="True", help="")
    parser.add_argument("--re_compute_emb", type=str, default="False", help="")
    return parser.parse_args()

def main():    
    args = parse_arguments()
    num_oracle = 10
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES")


    model_names = [
        args.model_name
    ]
    datasets = [
        args.dataset_name
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
                --flash_attn {args.flash_attn} \
                --model_name_or_path "{model_name}" \
                --dataset_dir data \
                --dataset "{dataset_name}" \
                --template default \
                --finetuning_type freeze \
                --output_dir "{output_dir}" \
                --overwrite_output_dir \
                --cutoff_len 1024 \
                --per_device_eval_batch_size 2 \
                --is_compute_emb {args.re_compute_emb} \
                --num_oracle "{num_oracle}"
            """
            run_cli_command(command)

    # wandb.log({"loss": 0})
    # wandb.finish()
if __name__ == "__main__":
    main()
