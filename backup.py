from src.llmtuner import run_exp

import json
import os
import copy
import numpy as np
from scipy.stats import entropy


def select_top_percent_questions_from_file(file_path, percent):
    question_scores = {}

    # Read data from JSONL file
    with open(file_path, 'r') as file:
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
    num_questions_to_select = int(num_questions * percent)

    # Get the top questions based on the specified percentage
    top_percent_questions = [question[0] for question in sorted_entropy[:num_questions_to_select]]

    return top_percent_questions

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

def run_cli_command(command):
    # Run the command
    os.system(command)


def main():
    model_name_or_path = "meta-llama/Llama-2-7b-hf" 
    dataset =  "arc_sample"
    dataset_dir  = "data"
    template = "default"
    finetuning_type  = "lora" 
    lora_target = "q_proj,v_proj" 
    cutoff_len = 1024 
    per_device_train_batch_size = 1 
    per_device_eval_batch_size = 1 
    gradient_accumulation_steps = 8 
    lr_scheduler_type = "cosine" 
    logging_steps = 10 
    save_steps = 100 
    eval_steps = 100 
    warmup_steps = 20
    evaluation_strategy = "steps" 
    learning_rate = 5e-5 
    num_train_epochs  = 1.0 
    max_samples = 2000
    val_size = 0.1 
    quantization_bit = 4 

    #### 
    num_iters = 5
    percentage = 0.1
    data_info_path = 'data/dataset_info.json'
    reward_model_path = "saves/llama2_7b/reward"
    dpo_adapter_path = "saves/llama2_7b/dpo"

    for iter in range(num_iters):
        print(f"{"="*20} ITERARION: {iter} {"="*20}")
        ##########################################################
        #### Inference
        ##########################################################
        print(f"{"-"*10} Inference {"-"*10}")
        batch_size_for_inference = 4
        if iter == 0:
            inference_command = f"""CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
                --stage rm \
                --do_predict \
                --model_name_or_path {model_name_or_path} \
                --dataset_dir {dataset_dir} \
                --dataset {dataset} \
                --template {template} \
                --output_dir {reward_model_path} \
                --per_device_eval_batch_size {batch_size_for_inference} \
                --quantization_bit {quantization_bit} \
                --fp16
                """
        else:
            inference_command = f"""CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
                --stage rm \
                --do_predict \
                --model_name_or_path {model_name_or_path} \
                --adapter_name_or_path {reward_model_path}\
                --finetuning_type {finetuning_type} \
                --lora_target {lora_target} \
                --dataset_dir {dataset_dir} \
                --dataset {dataset} \
                --template {template} \
                --output_dir {reward_model_path} \
                --per_device_eval_batch_size {batch_size_for_inference} \
                --quantization_bit {quantization_bit} \
                --fp16
                """
        
        run_cli_command(inference_command)

        print("Done inference")
        ##########################################################
        #### Selection
        ##########################################################
        print(f"{"="*10} Selection {"="*10}")

        prediction_path = f"{reward_model_path}/generated_predictions.jsonl"
        data_path = f"{dataset_dir}/{dataset}.json"
        output_file = f"{dataset_dir}/selected_entries.json"  # Name of the output file
        
        selected_questions = select_top_percent_questions_from_file(prediction_path, percentage)  # Select top 10% questions
        select_entries_by_ids(data_path, selected_questions, output_file)
        
        with open(data_info_path, 'r') as file:
            data_info = json.load(file)
            
            if dataset in data_info:
                # append new info to data_infor and store result in json file
                new_data_info = dataset + f"_iter_{iter}"
                data_info[new_data_info] = copy.deepcopy(data_info[dataset])
                data_info[new_data_info]["file_name"] = "selected_entries.json"

                with open(data_info_path, 'w') as outfile:
                    json.dump(data_info, outfile, indent=4)

                print("Updated dataset info has been stored in", data_info_path)

        print("Done selection")
        ##########################################################
        #### Train DPO
        ##########################################################
        print(f"{"-"*10} Train DPO {"-"*10}")

        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES
        
        dpo_ft_command = f"""CUDA_VISIBLE_DEVICES=0,1 accelerate launch\
            src/train_bash.py \
            --stage dpo \
            --do_train \
            --model_name_or_path {model_name_or_path} \
            --dataset_dir {dataset_dir} \
            --dataset {active_dataset} \
            --template {template} \
            --finetuning_type {finetuning_type} \
            --lora_target {lora_target} \
            --output_dir {dpo_adapter_path} \
            --overwrite_output_dir \
            --cutoff_len {cutoff_len} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size {per_device_train_batch_size} \
            --per_device_eval_batch_size {per_device_eval_batch_size} \
            --gradient_accumulation_steps {gradient_accumulation_steps} \
            --lr_scheduler_type {lr_scheduler_type} \
            --logging_steps {logging_steps} \
            --warmup_steps {warmup_steps} \
            --save_steps {save_steps} \
            --eval_steps {eval_steps} \
            --evaluation_strategy {evaluation_strategy} \
            --learning_rate {learning_rate} \
            --num_train_epochs {num_train_epochs} \
            --max_samples {max_samples} \
            --val_size {val_size} \
            --plot_loss \
            --dpo_ftx 1.0\
            --quantization_bit {quantization_bit}
            """

        run_cli_command(dpo_ft_command) 
        print(f"Done Train DPO {"-"*10}")
        ##########################################################
        #### Train Reward
        ##########################################################    
        print(f"{"-"*10} Train Reward {"-"*10}")
        active_dataset = new_data_info # replace dataset by ACTIVE QUERIES

        rm_ft_command = f"""CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            src/train_bash.py \
            --stage rm \
            --do_train \
            --flash_attn True\
            --model_name_or_path {model_name_or_path}\
            --adapter_name_or_path {dpo_adapter_path}\
            --output_dir {reward_model_path} \
            --dataset {active_dataset} \
            --dataset_dir {dataset_dir} \
            --template {template} \
            --finetuning_type {finetuning_type} \
            --lora_target {lora_target} \
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len {cutoff_len} \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size {per_device_train_batch_size} \
            --per_device_eval_batch_size {per_device_eval_batch_size} \
            --gradient_accumulation_steps {gradient_accumulation_steps} \
            --lr_scheduler_type {lr_scheduler_type} \
            --logging_steps {logging_steps} \
            --warmup_steps {warmup_steps} \
            --save_steps {save_steps} \
            --eval_steps {save_steps} \
            --evaluation_strategy {evaluation_strategy} \
            --learning_rate {learning_rate} \
            --num_train_epochs {num_train_epochs} \
            --max_samples {max_samples} \
            --val_size {val_size} \
            --ddp_timeout 1800000 \
            --plot_loss \
            --quantization_bit {quantization_bit}
            """

        run_cli_command(rm_ft_command) 
        print(f"Done train Reward {"-"*10}")
        ##########################################################
        #### Eval
        ########################################################## 
        task = "arc_easy"
        eval_output_path = f"saves/output_eval/iter_{iter}"
        eval_command = f"""lm_eval --model hf \
            --model_args pretrained={model_name_or_path},parallelize=True,load_in_4bit=True,peft={dpo_adapter_path} \
            --tasks {task} \
            --output_path {eval_output_path} \
            --device cuda:0 
            """

        # run_cli_command(eval_command)   
         

    print("DONE!!!")
if __name__ == "__main__":
    main()