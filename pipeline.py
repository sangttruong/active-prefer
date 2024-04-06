from src.llmtuner import run_exp

import argparse
import json
import random
import os
import copy

import numpy as np
from scipy.stats import entropy

def read_inference_results(prediction_path):
    question_scores = {}

    # Read inference result
    with open(prediction_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            question_id = entry["question"]
            chosen_score = entry["chosen"]
            rejected_scores = entry["rejected"]

            # check exits
            if question_id in question_scores:
                question_scores[question_id].append(rejected_scores)
            else:
                question_scores[question_id] = [chosen_score]

    return question_scores

def select_by_max_entropy(prediction_path, percent_or_num):
    question_scores = {}

    # Read inference result
    question_scores = read_inference_results(prediction_path)

    # Calculate the average of averages for each question
    entropy_vals = {}
    for question_id, scores in question_scores.items():
        softmax_scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
        entropy_value = entropy(softmax_scores, base=2)
        entropy_vals[question_id] = entropy_value

    # Sort the questions based on their average scores
    sorted_entropy = sorted(entropy_vals.items(), key=lambda x: x[1], reverse=True)

    # Calculate the number of questions to select based on the percentage
    num_questions = len(sorted_entropy)
    if percent_or_num < 1:
        num_questions_to_select = int(num_questions * percent_or_num)
    else:
        num_questions_to_select = min(num_questions, int(percent_or_num))
    
    selected_questions = [question[0] for question in sorted_entropy[:num_questions_to_select]]
    return selected_questions

def select_by_random(prediction_path, percent_or_num):
    question_scores = {}
    question_scores = read_inference_results(prediction_path)
    all_question_ids = list(set(question_scores.keys()))
    num_questions = len(all_question_ids)
    if percent_or_num < 1:
        num_questions_to_select = int(num_questions * percent_or_num)
    else:
        num_questions_to_select = min(num_questions, int(percent_or_num))
    selected_questions = random.sample(all_question_ids, num_questions_to_select)
    return selected_questions

def select_entries_by_ids(data_path, id_list, output_file):
    selected_entries = []

    # Read data from dataset.json
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = []
    for entry in data:
        if entry["id"] in id_list:
            selected_entries.append(entry)
        else:
            remaining_entries.append(entry)

    # Write remaining entries back to the original file
    with open(data_path, 'w') as file:
        json.dump(remaining_entries, file, indent=4)

    # Write selected entries to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(selected_entries, outfile, indent=4)

def delete_selected_info(data_info_path, pattern="iter"):
    # Read data from dataset.json
    with open(data_info_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = {}
    for name, v in data.items():
        if pattern not in name:
            remaining_entries[name] = v

    # Save the remaining data info
    with open(data_info_path, 'w') as outfile:
        json.dump(remaining_entries, outfile, indent=4)

    print("New data info with selected entries removed has been saved.")

def count_len_dataset(prediction_path):
    with open(prediction_path, 'r') as f:
        data = json.load(f)
        ids = set(item['id'] for item in data)
    return len(ids)


def run_cli_command(command):
    # Run the command
    os.system(command)

def main(args):
    # Prepare dataset

    if args.dataset_name in ['allenai/ai2_arc', 'arc']:
        prepare_data = f"""cd data/arc/ && python arc.py"""
        dataset = 'arc_challenge'
    elif args.dataset_name in ['truthful_qa']:
        prepare_data = f"""cd data/truthful_qa/ && python truthful_qa.py"""
        dataset = 'truthful_qa'
    elif args.dataset_name in ['Rowan/hellaswag', 'hellaswag']:
        prepare_data = f"""cd data/hellaswag/ && python hellaswag.py"""
        dataset = 'hellaswag'
    elif args.dataset_name in ['winogrande']:
        prepare_data = f"""cd data/winogrande/ && python winogrande.py"""
        dataset = 'winogrande'
    elif args.dataset_name in ['cais/mmlu', "mmlu"]:
        prepare_data = f"""cd data/mmlu/ && python mmlu.py"""
        dataset = 'mmlu'
    elif args.dataset_name in ['Anthropic/hh-rlhf', "hh-rlhf"]:
        prepare_data = f"""cd data/hh_rlhf/ && python hh_rlhf.py"""
        dataset = 'hh_rlhf'
    elif args.dataset_name in ['allenai/reward-bench', "reward-bench", "reward_bench"]:
        prepare_data = f"""cd data/reward_bench/ && python reward_bench.py"""
        dataset = 'reward_bench'
    else:
        raise(f"Does not support {args.dataset_name} dataset yet")

    print("Prepare Dataset")
    run_cli_command(prepare_data)

    model_name = args.model_name_or_path.split('/')[-1]
    reward_model_path = f"saves/{model_name}/{dataset}/reward"
    dpo_adapter_path = f"saves/{model_name}/{dataset}/dpo"
    
    num_sample_select = -1

    for iter in range(args.num_iters):
        print(f"ITERARION: {iter}")
        ##########################################################
        #### Inference
        ##########################################################
        print(f" Inference ")
        
        batch_size_for_inference = 4
        if iter == 0:
            inference_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_predict \
                --model_name_or_path {args.model_name_or_path} \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --output_dir {reward_model_path} \
                --per_device_eval_batch_size {batch_size_for_inference} \
                --quantization_bit {args.quantization_bit} \
                --fp16
                """
        else:
            inference_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_predict \
                --model_name_or_path {args.model_name_or_path} \
                --adapter_name_or_path {reward_model_path}\
                --finetuning_type {args.finetuning_type} \
                --lora_target {args.lora_target} \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --output_dir {reward_model_path} \
                --per_device_eval_batch_size {batch_size_for_inference} \
                --quantization_bit {args.quantization_bit} \
                --fp16
                """

        run_cli_command(inference_command)

        print("Done inference")
        ##########################################################
        #### Selection
        ##########################################################
        print(f" Selection ")

        prediction_path = f"{reward_model_path}/generated_predictions.jsonl"
        data_path = f"{args.dataset_dir}/{dataset}.json"
        output_file = f"{args.dataset_dir}/selected_entries.json"  # Name of the output file

        if num_sample_select == -1:
            num_sample_select = int(count_len_dataset(data_path) * args.percentage)

        # SELECTION METHOD
        if args.method == "max_entropy":
            selected_questions = select_by_max_entropy(prediction_path, num_sample_select)  
        elif args.method == "random":
            selected_questions = select_by_random(prediction_path, num_sample_select)  

        # Update data training
        select_entries_by_ids(data_path, selected_questions, output_file)
        with open(args.data_info_path, 'r') as file:
            data_info = json.load(file)
            
            if dataset in data_info:
                # append new info to data_infor and store result in json file
                new_data_info = dataset + f"_iter_{iter}"
                data_info[new_data_info] = copy.deepcopy(data_info[dataset])
                data_info[new_data_info]["file_name"] = "selected_entries.json"

                with open(args.data_info_path, 'w') as outfile:
                    json.dump(data_info, outfile, indent=4)

                print("Updated dataset info has been stored in", args.data_info_path)

        print("Done selection")
        ##########################################################
        #### Train DPO
        ##########################################################
        print(f" Train DPO ")

        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES
        
        dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch\
            --config_file examples/accelerate/fsdp_config.yaml \
            src/train_bash.py \
            --stage dpo \
            --do_train \
            --model_name_or_path {args.model_name_or_path} \
            --dataset_dir {args.dataset_dir} \
            --dataset {active_dataset} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --lora_target {args.lora_target} \
            --output_dir {dpo_adapter_path} \
            --overwrite_output_dir \
            --cutoff_len {args.cutoff_len} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size {args.per_device_train_batch_size} \
            --per_device_eval_batch_size {args.per_device_eval_batch_size} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} \
            --lr_scheduler_type {args.lr_scheduler_type} \
            --logging_steps {args.logging_steps} \
            --warmup_steps {args.warmup_steps} \
            --save_steps {args.save_steps} \
            --eval_steps {args.eval_steps} \
            --evaluation_strategy {args.evaluation_strategy} \
            --learning_rate {args.learning_rate} \
            --num_train_epochs {args.num_train_epochs} \
            --max_samples {args.max_samples} \
            --val_size {args.val_size} \
            --plot_loss \
            --dpo_ftx 1.0\
            --quantization_bit {args.quantization_bit}
            """

        run_cli_command(dpo_ft_command) 
        print("Done Train DPO")
        ##########################################################
        #### Train Reward
        ##########################################################    
        print(" Train Reward ")
        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES

        rm_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch \
            --config_file examples/accelerate/fsdp_config.yaml \
            src/train_bash.py \
            --stage rm \
            --do_train \
            --flash_attn True\
            --model_name_or_path {args.model_name_or_path}\
            --adapter_name_or_path {dpo_adapter_path}\
            --output_dir {reward_model_path} \
            --dataset {active_dataset} \
            --dataset_dir {args.dataset_dir} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --lora_target {args.lora_target} \
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len {args.cutoff_len} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size {args.per_device_train_batch_size} \
            --per_device_eval_batch_size {args.per_device_eval_batch_size} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} \
            --lr_scheduler_type {args.lr_scheduler_type} \
            --logging_steps {args.logging_steps} \
            --warmup_steps {args.warmup_steps} \
            --save_steps {args.save_steps} \
            --eval_steps {args.save_steps} \
            --evaluation_strategy {args.evaluation_strategy} \
            --learning_rate {args.learning_rate} \
            --num_train_epochs {args.num_train_epochs} \
            --max_samples {args.max_samples} \
            --val_size {args.val_size} \
            --ddp_timeout 1800000 \
            --plot_loss \
            --quantization_bit {args.quantization_bit}
            """

        run_cli_command(rm_ft_command) 
        print("Done train Reward ")
        ##########################################################
        #### Eval
        ########################################################## 
        if dataset == "arc_challenge":
            task_eval = "arc_challenge"
        elif dataset == "truthful_qa":
            task_eval = "truthfulqa_mc1"
        elif dataset == 'hellaswag':
            task_eval = "hellaswag"
        elif dataset == 'winogrande':
            task_eval = "winogrande"
        else:
            raise(f"Not support {dataset}")
   
        
        eval_output_path = f"saves/output_eval/iter_{iter}"
        adapter_path = f"../{dpo_adapter_path}"
        device = args.gpu_ids.split(",")[0]
        eval_command = f"""cd lm-evaluation-harness && lm_eval --model hf \
            --model_args pretrained={args.model_name_or_path},parallelize=True,load_in_4bit=True,peft={adapter_path} \
            --tasks {task_eval} \
            --output_path {eval_output_path} \
            --batch_size 16 \
            --limit 200 \
            --device cuda:{device}
            """
  
        run_cli_command(eval_command)  
        
        ########################################################## 
        # run_cli_command("cd ..") 
        print("=========================================================")
        
    print("DONE!!!")
    delete_selected_info(args.data_info_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--template", type=str, default="default", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LORA target")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--quantization_bit", type=int, default=4, help="Quantization bit")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")

    #######################
    parser.add_argument("--dataset_name", type=str, default="arc", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")    


    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of top questions to select")
    parser.add_argument("--method", type=str, default="max_entropy", help="Selection method")
    parser.add_argument("--dataset", type=str, default="arc_sample", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)