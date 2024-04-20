import numpy as np

from src.llmtuner import run_exp

import argparse
import json
import random
import os
import copy
import json
import math

from scipy.stats import entropy

from openai import OpenAI

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
    # Read data from dataset_info.json
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

def add_new_dataset_info(dataset_info_path, name, path):
    # Read data from dataset_info.json
    with open(dataset_info_path, 'r') as file:
        data = json.load(file)

    template = data['template']
    data[name] = copy.deepcopy(template)
    data[name]['file_name'] = path

    # Save new data info
    with open(dataset_info_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def delete_item_dataset_info(dataset_info_path, name):
    # Read data from dataset_info.json
    with open(dataset_info_path, 'r') as file:
        data = json.load(file)

    if name in data:
        # remove item in data
        del data[name]

    # Save new data info
    with open(dataset_info_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def jsonl_to_json(jsonl_file_path, output_json_file_path):
    # Read JSONL file
    with open(jsonl_file_path, 'r') as jsonl_file:
        lines = jsonl_file.readlines()

    # Parse each line as JSON and store in a list
    json_data = [json.loads(line.strip()) for line in lines]

    # Write the list of JSON objects to a JSON file
    with open(output_json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def count_len_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
        ids = set(item['id'] for item in data)
    return len(ids)

def run_cli_command(command):
    # Run the command
    os.system(command)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_accuracy(file_path):
    total_samples = 0
    correct_predictions = 0

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            chosen_value = sigmoid(data['chosen'])
            if chosen_value > 0.5:
                prediction = 1
            else:
                prediction = 0
            
            if prediction == 1:
                correct_predictions += 1
            
            total_samples += 1
    
    if total_samples == 0:
        return 0.0
    else:
        accuracy = correct_predictions / total_samples
        return accuracy

def save_eval_metric(file_path, accuracy, iteration):
    # Ensure the directory containing the file exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    json_data = [{'iteration': iteration, 'accuracy': accuracy}]

    with open(file_path, 'w') as file:
        json.dump(json_data, file) 

def delete_json_file(file_path):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # Remove the file
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        else:
            print(f"File '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def copy_json_file(source_path, destination_path):
    try:
        # Open the JSON file from the source path
        with open(source_path, 'r') as source_file:
            # Load JSON data
            data = json.load(source_file)
        
        # Write the JSON data to the destination path
        with open(destination_path, 'w') as dest_file:
            json.dump(data, dest_file, indent=4)
        
        print(f"JSON file copied from '{source_path}' to '{destination_path}' successfully.")
    except FileNotFoundError:
        print("One or both paths are invalid.")
    except Exception as e:
        print(f"An error occurred: {e}")


import subprocess
import time

def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(cmd_string, shell=True)
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def shutdown_server(process):
    try:
        process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")

   
    
    


def main(args):
    # Prepare dataset
    source_path = f"{args.dataset_dir}/backup_dataset_info.json"
    destination_path = f"{args.dataset_dir}/dataset_info.json"
    copy_json_file(source_path, destination_path)
    delete_json_file(f"{args.dataset_dir}/selected_entries.json")

    if args.dataset_name in ['allenai/ai2_arc', 'arc', "arc_challenge_train"]:
        dataset = 'arc_challenge_train'
        prepare_data = f"""python data/arc/arc.py --sanity_check {args.sanity_check}"""
    elif args.dataset_name in ['truthful_qa', "truthful_qa_train"]:
        dataset = 'truthful_qa_train'
        prepare_data = f"""python data/truthful_qa/truthful_qa.py --sanity_check {args.sanity_check}"""
    elif args.dataset_name in ['Rowan/hellaswag', 'hellaswag', "hellaswag_train"]:
        prepare_data = f"""python data/hellaswag/hellaswag.py --sanity_check {args.sanity_check}"""
        dataset = 'hellaswag_train'
    elif args.dataset_name in ['winogrande', "winogrande_train"]:
        prepare_data = f"""python data/winogrande/winogrande.py --sanity_check {args.sanity_check}"""
        dataset = 'winogrande_train'
    elif args.dataset_name in ['cais/mmlu', "mmlu", "mmlu_train"]:
        prepare_data = f"""python data/mmlu/mmlu.py --sanity_check {args.sanity_check}"""
        dataset = 'mmlu_train'
    elif args.dataset_name in ['Anthropic/hh-rlhf', "hh-rlhf", "hh_rlhf" "hh_rlhf_train"]:
        prepare_data = f"""python data/hh_rlhf/hh_rlhf.py"""
        dataset = 'hh_rlhf_train'
    elif args.dataset_name in ['allenai/reward-bench', "reward-bench", "reward_bench", "reward_bench_train"]:
        prepare_data = f"""python data/reward_bench/reward_bench.py --sanity_check {args.sanity_check}"""
        dataset = 'reward_bench_train'
    else:
        raise(f"Does not support {args.dataset_name} dataset yet")

    testset = dataset.replace("train", "test")
    print(f"Prepare Dataset: {prepare_data} .....................")
    run_cli_command(prepare_data)


    model_name = args.model_name_or_path.split('/')[-1]
    reward_model_path = f"saves/{model_name}/{dataset}/{args.method}/reward"
    dpo_adapter_path = f"saves/{model_name}/{dataset}/{args.method}/dpo"
    oracle_adapter_path = f"saves/{model_name}/{dataset}/{args.method}/oracle"
    eval_metric_dir = f"saves/{model_name}/{dataset}/{args.method}"
    num_sample_selected = int(count_len_dataset(f"{args.dataset_dir}/{dataset}.json") * args.percentage)

    # Train an Oracle model O 
    ft_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
        --stage oracle \
        --do_train \
        --flash_attn True\
        --model_name_or_path {args.model_name_or_path}\
        --output_dir {oracle_adapter_path}\
        --dataset {dataset} \
        --dataset_dir {args.dataset_dir} \
        --template {args.template} \
        --finetuning_type full \
        --overwrite_output_dir \
        --cutoff_len {args.cutoff_len} \
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
        --val_size 0.1 \
        --ddp_timeout 1800000 \
        --plot_loss 
        """
    
    print(f"Training Oracle model ............................")
    # run_cli_command(ft_oracle_command)
    

    # active pipeline     
    for iter in range(args.num_iters):
        print(f"######### ITERARION: {iter} ############")
        ##########################################################
        #### SELECTION
        ##########################################################
        print(f"Selection ........................")
        # active_dataset = f"{dataset}_iter_{iter}" # replace dataset by ACTIVE QUERIES

        if args.use_accelerate_eval:
            selection_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port}\
                --config_file examples/accelerate/default.yaml \
                src/train_bash.py \
                --stage selection \
                --do_predict \
                --active_iter {iter}\
                --num_sample_selected {num_sample_selected} \
                --model_name_or_path {args.model_name_or_path} \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --finetuning_type full \
                --output_dir {reward_model_path} \
                --overwrite_output_dir \
                --cutoff_len {args.cutoff_len} \
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
                --num_train_epochs {args.num_train_epochs}
            """
        else: 
            selection_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py\
                --stage selection \
                --do_predict \
                --active_iter {iter}\
                --model_name_or_path {args.model_name_or_path} \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --finetuning_type full \
                --output_dir {reward_model_path} \
                --overwrite_output_dir \
                --cutoff_len {args.cutoff_len} \
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
                --num_train_epochs {args.num_train_epochs}\
                --num_sample_selected {num_sample_selected}
            """
        run_cli_command(selection_command) 
        ##########################################################
        #### TRAIN DPO
        ##########################################################
        print(f"Train DPO .................................")

        active_dataset = f"{dataset}_iter_{iter}" # replace dataset by ACTIVE QUERIES
        
        if args.use_accelerate: 
            dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port}\
                --config_file examples/accelerate/default.yaml \
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
                --per_device_train_batch_size {args.per_device_train_batch_size} \
                --gradient_accumulation_steps {args.gradient_accumulation_steps} \
                --lr_scheduler_type {args.lr_scheduler_type} \
                --logging_steps {args.logging_steps} \
                --warmup_steps {args.warmup_steps} \
                --save_steps {args.save_steps} \
                --learning_rate {args.learning_rate} \
                --num_train_epochs {args.num_train_epochs} \
                --max_samples {args.max_samples} \
                --plot_loss \
                --dpo_ftx 1.0 \
                --report_to none \
                --fp16
                """
        else:
            dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py\
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
                --gradient_accumulation_steps {args.gradient_accumulation_steps} \
                --lr_scheduler_type {args.lr_scheduler_type} \
                --logging_steps {args.logging_steps} \
                --warmup_steps {args.warmup_steps} \
                --save_steps {args.save_steps} \
                --learning_rate {args.learning_rate} \
                --num_train_epochs {args.num_train_epochs} \
                --max_samples {args.max_samples} \
                --plot_loss \
                --dpo_ftx 1.0
            """

        run_cli_command(dpo_ft_command) 
        # ----------------------------------------------------------------

        eval_dpo_ft_output_path = f"{dpo_adapter_path}/iter_{iter}"

        if args.use_accelerate_eval: 
            eval_dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port}\
                --config_file examples/accelerate/default.yaml \
                src/train_bash.py \
                --stage dpo \
                --do_eval \
                --model_name_or_path {args.model_name_or_path} \
                --adapter_name_or_path {dpo_adapter_path}
                --dataset_dir {args.dataset_dir} \
                --dataset {active_dataset} \
                --template {args.template} \
                --finetuning_type {args.finetuning_type} \
                --lora_target {args.lora_target} \
                --output_dir {eval_dpo_ft_output_path} \
                --cutoff_len {args.cutoff_len} \
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
                --dpo_ftx 1.0 \
                --report_to none \
                --fp16
                """
        else:
            eval_dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py\
                --stage dpo \
                --do_eval\
                --model_name_or_path {args.model_name_or_path} \
                --adapter_name_or_path {dpo_adapter_path}
                --dataset_dir {args.dataset_dir} \
                --dataset {active_dataset} \
                --template {args.template} \
                --finetuning_type {args.finetuning_type} \
                --lora_target {args.lora_target} \
                --output_dir {eval_dpo_ft_output_path} \
                --cutoff_len {args.cutoff_len} \
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
                --dpo_ftx 1.0
            """

        # print("Eval DPO  ..................................")
        # run_cli_command(eval_dpo_ft_command)

        ##########################################################
        #### Train Reward
        ##########################################################    
        print("Train Reward ..................................")

        if args.use_accelerate:
            rm_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
                --config_file examples/accelerate/default.yaml \
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
                --gradient_accumulation_steps {args.gradient_accumulation_steps} \
                --lr_scheduler_type {args.lr_scheduler_type} \
                --logging_steps {args.logging_steps} \
                --warmup_steps {args.warmup_steps} \
                --save_steps {args.save_steps} \
                --learning_rate {args.learning_rate} \
                --num_train_epochs {args.num_train_epochs} \
                --max_samples {args.max_samples} \
                --ddp_timeout 1800000 \
                --plot_loss \
                --only_training_vhead True \
                --report_to none \
                --fp16
                """
        else:
            rm_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
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
                --ddp_timeout 1800000 \
                --plot_loss \
                --only_training_vhead True\
                --report_to none\
                --fp16
                """

        run_cli_command(rm_ft_command) 
        # ----------------------------------------------------------------
        eval_rm_command_output_path = f"{reward_model_path}/iter_{iter}"

        if args.use_accelerate_eval:
            eval_rm_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
                --config_file examples/accelerate/default.yaml \
                src/train_bash.py \
                --stage rm \
                --flash_attn True\
                --model_name_or_path {args.model_name_or_path}\
                --adapter_name_or_path {reward_model_path}\
                --output_dir {eval_rm_command_output_path} \
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
                --ddp_timeout 1800000 \
                --plot_loss \
                --only_training_vhead True \
                --report_to none \
                --fp16
                """
        else:
            eval_rm_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_eval \
                --flash_attn True\
                --model_name_or_path {args.model_name_or_path}\
                --adapter_name_or_path {reward_model_path}\
                --output_dir {eval_rm_command_output_path} \
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
                --ddp_timeout 1800000 \
                --plot_loss \
                --only_training_vhead True\
                --report_to none\
                --fp16
                """

        # print("Eval Reward ..................................")
        # run_cli_command(eval_rm_command) 
   
        ##########################################################
        ####  Eval
        ##########################################################

        print(f"Generating text from the new DPO model .....................")
        dataset_name_generated = f"{dataset}_generated"

        # if args.use_accelerate_eval:
        #     generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        #         --config_file examples/accelerate/default.yaml \
        #         src/train_bash.py \
        #         --stage sft \
        #         --do_predict \
        #         --model_name_or_path {args.model_name_or_path} \
        #         --adapter_name_or_path {dpo_adapter_path} \
        #         --dataset {testset} \
        #         --dataset_dir {args.dataset_dir} \
        #         --template {args.template} \
        #         --finetuning_type {args.finetuning_type} \
        #         --output_dir {args.dataset_dir} \
        #         --overwrite_cache \
        #         --overwrite_output_dir \
        #         --cutoff_len {args.cutoff_len} \
        #         --preprocessing_num_workers 16 \
        #         --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        #         --predict_with_generate \
        #         --report_to none \
        #         --fp16
        #     """
        # else:
        #     generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
        #         --stage sft \
        #         --do_predict \
        #         --model_name_or_path {args.model_name_or_path} \
        #         --adapter_name_or_path {dpo_adapter_path} \
        #         --dataset {testset} \
        #         --dataset_dir {args.dataset_dir} \
        #         --template {args.template} \
        #         --finetuning_type {args.finetuning_type} \
        #         --output_dir {args.dataset_dir} \
        #         --overwrite_cache \
        #         --overwrite_output_dir \
        #         --cutoff_len {args.cutoff_len} \
        #         --preprocessing_num_workers 16 \
        #         --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        #         --predict_with_generate \
        #         --fp16
        #     """
        # add dataset_name_generated into dataset_info to inference oracle model
        # jsonl_to_json(f"{args.dataset_dir}/generated_predictions.jsonl", f"{args.dataset_dir}/generated_predictions.json")
        # add_new_dataset_info(args.data_info_path, dataset_name_generated, f"generated_predictions.json")

        

        # Export DPO finetuned model 
        dpo_full_path = f"{dpo_adapter_path}/full"

        export_command = f""""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/export_model.py \
            --model_name_or_path {args.model_name_or_path} \
            --adapter_name_or_path {dpo_adapter_path} \
            --export_hub_model_id
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --export_dir {dpo_full_path} \
            --export_size 2 \
            --export_legacy_format False
        """
        run_cli_command(export_command)

        # Push model to hf-hub
        
        # Deploy
        api_port = 8006
        if "llama" in args.model_name_or_path.lower():
            template = "llama"
        elif "mistral" in args.model_name_or_path.lower():
            template = "mistral"

        deploy_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} API_PORT={args.api_port} python src/api_demo.py \
            --model_name_or_path {dpo_full_path}\
            --template {template} \
            --infer_backend vllm \
            --vllm_enforce_eager
        """
        server_process = run_server(deploy_command)
        # Inference 

        
        client = OpenAI(
            base_url=f"http://localhost:{args.api_port}/v1",
            api_key="token-abc123",
        )

        testset_path = f"{args.dataset_dir}/{testset}.json" 
        with open(testset_path, 'r') as json_file:
            test_data = json.load(json_file)

        predictions = []
        for sample in test_data:
            pred_sample = copy.deepcopy(sample)

            completion = client.chat.completions.create(
                model=dpo_full_path,
                messages=[
                    {"role": "user", "content": pred_sample['instruction']}
                ]
            )
            pred = completion.choices[0].message.content
            pred_sample['output'][0] = pred
            
            predictions.append(pred_sample)


        # Save result at "args.dataset_dir}/generated_predictions.json" 
        output_file_path = os.path.join(args.dataset_dir, "generated_predictions.json")

        # Save the predictions to the JSON file
        with open(output_file_path, 'w') as output_file:
            json.dump(predictions, output_file)
        print(f"Predictions saved to: {output_file_path}")
        
        # Add new dataset info to datset_info.json to run predict reward model
        add_new_dataset_info(args.data_info_path, dataset_name_generated, f"generated_predictions.json")

        # Shutdown 
        if server_process:
            try:
                # Shutdown the server
                shutdown_server(server_process)
            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Shutting down server.")
                shutdown_server(server_process)

        # -------------
        if args.use_accelerate_eval:
            inference_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
                --config_file examples/accelerate/default.yaml \
                src/train_bash.py \
                --stage rm \
                --do_predict \
                --flash_attn True\
                --model_name_or_path {args.model_name_or_path} \
                --vhead_oracle_path {oracle_adapter_path}\
                --finetuning_type full \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset_name_generated} \
                --template {args.template} \
                --output_dir {oracle_adapter_path} \
                --per_device_eval_batch_size {args.per_device_eval_batch_size} \
                --fp16
                """
        else:
            inference_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_predict \
                --flash_attn True\
                --model_name_or_path {args.model_name_or_path} \
                --vhead_oracle_path {oracle_adapter_path}\
                --finetuning_type full \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset_name_generated} \
                --template {args.template} \
                --output_dir {oracle_adapter_path} \
                --per_device_eval_batch_size {args.per_device_eval_batch_size} \
                --fp16
                """
            
        print(f"Inference Oracle model ............................")
        run_cli_command(inference_oracle_command)
        delete_item_dataset_info(args.data_info_path, dataset_name_generated) # clean dataset_info

        # -------------------------
        # Get accuracy        
        accuracy = calculate_accuracy(f"{oracle_adapter_path}/generated_predictions.jsonl")
        print("Accuracy:", accuracy)
        
        save_eval_metric_path = f"{eval_metric_dir}/accuracy_results_{iter}.json"
        save_eval_metric(save_eval_metric_path, accuracy, iter)

        print("=========================================================")
        
    print("DONE!!!")
    delete_selected_info(args.data_info_path, f"{model_name}_iter")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--template", type=str, default="default", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LORA target")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=400, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=400, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--quantization_bit", type=int, default=4, help="Quantization bit")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--sanity_check", action="store_true", help="Test")
    parser.add_argument("--use_accelerate", action="store_true", help="is using accelerate")
    parser.add_argument("--use_accelerate_eval", action="store_true", help="is using accelerate")

    #######################
    parser.add_argument("--dataset_name", type=str, default="arc", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")    

    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of top questions to select")
    parser.add_argument("--method", type=str, default="max_entropy", help="Selection method")
    parser.add_argument("--dataset", type=str, default="arc_sample", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29505, help="Deepspeed Port")
    parser.add_argument("--api_port", type=int, default=8005, help="Deploy API port")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)  