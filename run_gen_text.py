import os
import argparse
import wandb



def run_cli_command(command):
    # Run the command
    print(command)
    os.system(command)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--dataset_name", type=str, default="False", help="Test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Test")
    return parser.parse_args()

def main():
    wandb.init(project="oracle")

    # model_names = [
    #     "mistralai/Mistral-7B-v0.1",
    #     "mistralai/Mistral-7B-Instruct-v0.2",
    #     "google/gemma-7b",
    #     "google/gemma-7b-it",
    #     "meta-llama/Llama-2-7b-hf",
    #     "meta-llama/Llama-2-7b-chat-hf",
    #     "meta-llama/Llama-2-13b-hf",
    #     "meta-llama/Llama-2-13b-chat-hf",
    #     "meta-llama/Meta-Llama-3-8B",
    #     "meta-llama/Meta-Llama-3-8B-Instruct"
    # ]
    # datasets = [
    #     "arc_challenge", 
    #     "truthful_qa",
    #     'winogrande',
        # "reward_bench",
        # "hellaswag",
        # "mmlu",
        # "hh_rlhf",
    # ]

    args = parse_arguments()
    num_iter = 100
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES")
 
    for i in range(num_iter):
        model_name_short = args.model_name.split('/')[-1]
        dataset_name = f"{args.dataset_name}_train_Llama-2-7b-hf_random"
        output_dir = f"saves/{model_name_short}/{args.dataset_name}/text/gen_text_{i}"

        # Training command
        command = f"""CUDA_VISIBLE_DEVICES={gpu_device} accelerate launch \
            --config_file examples/accelerate/single_config.yaml \
            src/train_bash.py \
            --stage sft \
            --do_predict \
            --model_name_or_path {args.model_name}\
            --dataset {dataset_name} \
            --dataset_dir data \
            --template default \
            --finetuning_type freeze \
            --output_dir {output_dir}\
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len 1024 \
            --per_device_eval_batch_size 2\
            --temperature 1.0\
            --top_p 1\
            --top_k 50\
            --predict_with_generate
        """
        run_cli_command(command)

    wandb.log({"loss": 0})
    wandb.finish()
if __name__ == "__main__":
    main()
