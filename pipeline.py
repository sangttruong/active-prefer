from src.llmtuner import run_exp

import argparse
import json
import random
import os
import copy
import json
import math

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




def count_len_dataset(prediction_path):
    with open(prediction_path, 'r') as f:
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
    breakpoint()
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        if json_data: 
            json_data.append({'iteration': iteration, 'accuracy': accuracy})
        else:
            json_data = [{'iteration': iteration, 'accuracy': accuracy}]

        with open(file_path, 'w') as file:
            json.dump(json_data, file)
   


def main(args):
    # Prepare dataset

    if args.dataset_name in ['allenai/ai2_arc', 'arc']:
        dataset = 'arc_challenge_train'
        prepare_data = f"""python data/arc/arc.py --sanity_check {args.sanity_check}"""
    elif args.dataset_name in ['truthful_qa']:
        dataset = 'truthful_qa_train'
        prepare_data = f"""python data/truthful_qa/truthful_qa.py --sanity_check {args.sanity_check}"""
    elif args.dataset_name in ['Rowan/hellaswag', 'hellaswag']:
        prepare_data = f"""python data/hellaswag/hellaswag.py --sanity_check {args.sanity_check}"""
        dataset = 'hellaswag_train'
    elif args.dataset_name in ['winogrande']:
        prepare_data = f"""python data/winogrande/winogrande.py --sanity_check {args.sanity_check}"""
        dataset = 'winogrande_train'
    elif args.dataset_name in ['cais/mmlu', "mmlu"]:
        prepare_data = f"""python data/mmlu/mmlu.py --sanity_check {args.sanity_check}"""
        dataset = 'mmlu_train'
    elif args.dataset_name in ['Anthropic/hh-rlhf', "hh-rlhf"]:
        prepare_data = f"""cd data/hh_rlhf/ && python hh_rlhf.py"""
        dataset = 'hh_rlhf'
    elif args.dataset_name in ['allenai/reward-bench', "reward-bench", "reward_bench"]:
        prepare_data = f"""cd data/reward_bench/ && python reward_bench.py"""
        dataset = 'reward_bench'
    else:
        raise(f"Does not support {args.dataset_name} dataset yet")

    testset = dataset.replace("train", "test")
    print(f"Prepare Dataset: {prepare_data} .....................")
    run_cli_command(prepare_data)

    num_sample_select = -1 

    model_name = args.model_name_or_path.split('/')[-1]
    reward_model_path = f"saves/{model_name}/{dataset}/{args.method}/reward"
    dpo_adapter_path = f"saves/{model_name}/{dataset}/{args.method}/dpo"
    oracle_adapter_path = f"saves/{model_name}/{dataset}/{args.method}/oracle"
    eval_metric_path = f"saves/{model_name}/{dataset}/accuracy_results.json"

    
    # Train an Oracle model O on 80% of the data
    ft_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
        --stage rm \
        --do_train \
        --do_eval \
        --flash_attn True\
        --model_name_or_path {args.model_name_or_path}\
        --output_dir {oracle_adapter_path}\
        --dataset {dataset} \
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
        --val_size 0.8 \
        --ddp_timeout 1800000 \
        --plot_loss \
        --quantization_bit {args.quantization_bit}\
        --only_training_vhead False
        """
    print(f"Training Oracle model ............................")
    run_cli_command(ft_oracle_command)

    # active pipeline     
    for iter in range(args.num_iters):
        print(f"######### ITERARION: {iter} ############")
        ##########################################################
        #### Inference
        ##########################################################
        print(f"Inference ..............")
        
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
                --do_eval \
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

        
        ##########################################################
        #### Selection
        ##########################################################
        print(f"Selection ........................")

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
                new_data_info = f"{dataset}_{model_name}_iter_{iter}"
                data_info[new_data_info] = copy.deepcopy(data_info[dataset])
                data_info[new_data_info]["file_name"] = "selected_entries.json"

                with open(args.data_info_path, 'w') as outfile:
                    json.dump(data_info, outfile, indent=4)

                print("Updated dataset info has been stored in", args.data_info_path)

        ##########################################################
        #### Train DPO
        ##########################################################
        print(f"Train DPO .................................")

        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES
        
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
        else:
            dpo_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py\
                --stage dpo \
                --do_train \
                --do_eval\
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
        ##########################################################
        #### Train Reward
        ##########################################################    
        print("Train Reward ..................................")
        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES

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
                --quantization_bit {args.quantization_bit}\
                --only_training_vhead True
                """
        else:
            rm_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_train \
                --do_eval \
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
                --quantization_bit {args.quantization_bit}\
                --only_training_vhead True
                """

        run_cli_command(rm_ft_command) 
   
        ##########################################################
        ####  Eval
        ##########################################################
        dataset_name_generated = f"{dataset}_generated"

        generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
            --stage sft \
            --do_predict \
            --model_name_or_path {args.model_name_or_path} \
            --adapter_name_or_path {dpo_adapter_path} \
            --dataset {testset} \
            --dataset_dir {args.dataset_dir} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --output_dir {args.dataset_dir} \
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len {args.cutoff_len} \
            --preprocessing_num_workers 16 \
            --per_device_eval_batch_size {args.per_device_eval_batch_size} \
            --max_samples 10 \
            --predict_with_generate \
            --fp16
        """

        print(f"Generating text from the new DPO model .....................")
        run_cli_command(generate_text_command)

        
        # add dataset_name_generated into dataset_info to inference oracle model
        jsonl_to_json(f"{args.dataset_dir}/generated_predictions.jsonl", f"{args.dataset_dir}/generated_predictions.json")
        add_new_dataset_info(args.data_info_path, dataset_name_generated, f"generated_predictions.json")


        inference_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage rm \
                --do_predict \
                --flash_attn True\
                --model_name_or_path {args.model_name_or_path} \
                --adapter_name_or_path {oracle_adapter_path}\
                --finetuning_type {args.finetuning_type} \
                --lora_target {args.lora_target} \
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset_name_generated} \
                --template {args.template} \
                --output_dir {oracle_adapter_path} \
                --per_device_eval_batch_size {batch_size_for_inference} \
                --quantization_bit {args.quantization_bit} \
                --fp16
                """
        
        print(f"Inference Oracle model ............................")
        run_cli_command(inference_oracle_command)
        delete_item_dataset_info(args.data_info_path, dataset_name_generated) # clean dataset_info

        # Get accuracy        
        accuracy = calculate_accuracy(f"{oracle_adapter_path}/generated_predictions.jsonl")
        print("Accuracy:", accuracy)
        save_eval_metric(eval_metric_path, accuracy, iter)

        print("=========================================================")
        
    print("DONE!!!")
    delete_selected_info(args.data_info_path, f"{model_name}_iter")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--template", type=str, default="default", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LORA target")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--quantization_bit", type=int, default=4, help="Quantization bit")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--sanity_check", action="store_true", help="Test")
    parser.add_argument("--use_accelerate", type=bool, default=False, help="is using accelerate")

    #######################
    parser.add_argument("--dataset_name", type=str, default="arc", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")    


    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of top questions to select")
    parser.add_argument("--method", type=str, default="max_entropy", help="Selection method")
    parser.add_argument("--dataset", type=str, default="arc_sample", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29500, help="Port")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)