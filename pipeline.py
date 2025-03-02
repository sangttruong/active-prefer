import numpy as np

from src.llmtuner import run_exp

import argparse
import json
import random
import os
import subprocess
import time
import copy
import json
import math
from tqdm import tqdm

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

    if name in data:
        del data[name]  # Remove the existing entry if it exists

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


def find_and_kill_process(command):
    # Construct the command to find the PID
    find_pid_command = f"""pgrep -af "{command}" """

    # Execute the command to find the PID
    pid_output = subprocess.check_output(find_pid_command, shell=True)

    # Decode the output and split it into lines
    pid_lines = pid_output.decode().splitlines()

    # Extract the PID(s)
    pids = [line.split()[0] for line in pid_lines]

    print("PID(s) of the process:")
    print(pids)

    # Construct the command to kill the process(es)
    if pids:
        kill_pid_command = f"kill -9 {' '.join(pids)}"

        # Execute the command to kill the process(es)
        subprocess.run(kill_pid_command, shell=True)
        print("Process(es) killed.")
    else:
        print("No matching process found.")

def main(args):

    model_name = args.model_name_or_path.split('/')[-1]

    if args.dataset_name in ['allenai/ai2_arc', 'arc', "arc_challenge_train"]:
        prepare_data = f"""python data/arc/arc.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'arc_challenge_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['truthful_qa', "truthful_qa_train"]:
        prepare_data = f"""python data/truthful_qa/truthful_qa.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'truthful_qa_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['Rowan/hellaswag', 'hellaswag', "hellaswag_train"]:
        prepare_data = f"""python data/hellaswag/hellaswag.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'hellaswag_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['winogrande', "winogrande_train"]:
        prepare_data = f"""python data/winogrande/winogrande.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'winogrande_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['cais/mmlu', "mmlu", "mmlu_train"]:
        prepare_data = f"""python data/mmlu/mmlu.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'mmlu_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['Anthropic/hh-rlhf', "hh-rlhf", "hh_rlhf" "hh_rlhf_train"]:
        prepare_data = f"""python data/hh_rlhf/hh_rlhf.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'hh_rlhf_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    elif args.dataset_name in ['allenai/reward-bench', "reward-bench", "reward_bench", "reward_bench_train"]:
        prepare_data = f"""python data/reward_bench/reward_bench.py --sanity_check {args.sanity_check} --model_name {model_name} --method {args.method}"""
        dataset = f'reward_bench_train_{model_name}_{args.method}{"_check" if args.sanity_check else ""}'
    else:
        raise(f"Does not support {args.dataset_name} dataset yet")

    testset = dataset.replace("train", "test")
    print(f"Prepare Dataset: {prepare_data} .....................")
    run_cli_command(prepare_data)


    reward_model_path = f"saves/{model_name}/{dataset}/{args.method}/reward"
    dpo_adapter_path = f"saves/{model_name}/{dataset}/{args.method}/dpo"
    oracle_adapter_path = f"saves/{model_name}/{dataset}/oracle{'_check' if args.sanity_check else ''}"
    eval_metric_dir = f"saves/{model_name}/{dataset}/{args.method}"
    num_sample_selected = int(count_len_dataset(f"{args.dataset_dir}/{dataset}.json") * args.percentage)

    # Train an Oracle model O 
    if args.is_retrain_oracle == False and os.path.exists(oracle_adapter_path):
        print("Oracle trained")
    elif args.type_of_oracle == 'lora':
        ft_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
            --stage rm  \
            --do_train \
            --flash_attn {args.flash_attn}\
            --model_name_or_path {args.model_name_or_path}\
            --output_dir {oracle_adapter_path}\
            --dataset {dataset} \
            --dataset_dir {args.dataset_dir} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --lora_target {args.lora_target} \
            --overwrite_output_dir \
            --cutoff_len {args.cutoff_len} \
            --per_device_train_batch_size {args.per_device_train_batch_size} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} \
            --lr_scheduler_type {args.lr_scheduler_type} \
            --logging_steps {args.logging_steps} \
            --warmup_steps {args.warmup_steps} \
            --save_steps {args.save_steps} \
            --learning_rate {args.learning_rate} \
            --num_train_epochs {5*args.num_train_epochs}\
            --max_samples {args.max_samples} \
            --is_compute_emb {args.is_retrain_oracle}\
            --plot_loss \
            --fp16
            """
        print(f"Training Oracle model ............................")
        run_cli_command(ft_oracle_command)
    else:
        ft_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py\
            --stage oracle \
            --do_train \
            --flash_attn {args.flash_attn}\
            --model_name_or_path {args.model_name_or_path}\
            --output_dir {oracle_adapter_path}\
            --dataset {dataset} \
            --dataset_dir {args.dataset_dir} \
            --template {args.template} \
            --finetuning_type freeze\
            --overwrite_output_dir \
            --cutoff_len {args.cutoff_len} \
            --per_device_eval_batch_size {args.per_device_eval_batch_size} \
            --is_compute_emb {args.is_retrain_oracle}\
            --num_oracle 1\
            --plot_loss \
            --fp16
            """
        print(f"Training Oracle model ............................")
        run_cli_command(ft_oracle_command)
    
    active_accuracy = []
    # active pipeline     
    for iter in range(args.num_iters):
        print(f"######### ITERARION: {iter} ############")
        ##########################################################
        #### SELECTION
        ##########################################################
        print(f"Selection ........................")
        if args.method in ['max_entropy', "random", "least_conf"]:
            selection_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage selection \
                --do_predict \
                --model_name_or_path {args.model_name_or_path} \
                {'--adapter_name_or_path ' + reward_model_path if iter != 0 else ''}\
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --finetuning_type freeze \
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
                --num_train_epochs {3*args.num_train_epochs}\
                --active_iter {iter}\
                --acquisition {args.method}\
                --num_sample_selected {num_sample_selected}
            """
            run_cli_command(selection_command) 
        elif args.method in ['qbc', 'bald']:
            selection_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage selection \
                --do_predict \
                --model_name_or_path {args.model_name_or_path} \
                {'--adapter_name_or_path ' + reward_model_path if iter != 0 else ''}\
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset} \
                --template {args.template} \
                --finetuning_type freeze \
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
                --num_train_epochs {3*args.num_train_epochs}\
                --active_iter {iter}\
                --acquisition {args.method}\
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
                {'--adapter_name_or_path ' + dpo_adapter_path if iter != 0 else ''}\
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

        run_cli_command(dpo_ft_command) 
        # ------------------------------------------------------------------- 
        # Export DPO finetuned model 
        dpo_full_path = f"{dpo_adapter_path}/full"

        export_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/export_model.py \
            --model_name_or_path {args.model_name_or_path} \
            --adapter_name_or_path {dpo_adapter_path} \
            --export_dir {dpo_full_path} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --export_size 2 \
            --export_legacy_format False
            """
        
        # Export model
        print(f"Merge LoRA.............")
        run_cli_command(export_command) 
        # ==================================================================
        
        ##########################################################
        #### UPDATE SELECTOR
        ##########################################################    
        print("Train Reward ..................................")
        if args.method in ['max_entropy', "random"]:
            if args.use_accelerate:
                rm_ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
                    --config_file examples/accelerate/default.yaml \
                    src/train_bash.py \
                    --stage rm \
                    --do_train \
                    --flash_attn {args.flash_attn}\
                    --model_name_or_path {dpo_full_path}\
                    --output_dir {reward_model_path} \
                    --dataset {active_dataset} \
                    --dataset_dir {args.dataset_dir} \
                    --template {args.template} \
                    --finetuning_type freeze \
                    --overwrite_cache \
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
                    --plot_loss \
                    --report_to none \
                    --fp16
                    """

            run_cli_command(rm_ft_command) 
        elif args.method in ['qbc']:
            print("Updatse commitees")
        # ----------------------------------------------------------------
        ##########################################################
        ####  Eval
        ##########################################################
        print(f"Generating text from the new DPO model .....................")
        dataset_name_generated = f"{dataset}_generated"
       

        if args.is_using_vllm:
            # Deploy
            if "llama" in args.model_name_or_path.lower():
                template = "llama2"
            elif "mistral" in args.model_name_or_path.lower():
                template = "mistral"

            deploy_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} API_PORT={args.api_port} python src/api_demo.py --model_name_or_path {dpo_full_path} --template {template} --infer_backend vllm --vllm_enforce_eager"""

            print("Deploy LLM.....")
            server_process = run_server(deploy_command) # subprocess
            time.sleep(60)  # Sleep for 60 seconds
            print("OKE")

            # Inference 
            client = OpenAI(
                base_url=f"http://localhost:{args.api_port}/v1",
                api_key="token-abc123",
            )

            testset_path = f"{args.dataset_dir}/{testset}.json" 
            with open(testset_path, 'r') as json_file:
                test_data = json.load(json_file)

            predictions = []
            for sample in tqdm(test_data):
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

            # Shutdown 
            kill_cmd = f"python src/api_demo.py --model_name_or_path {dpo_full_path} --template {template} --infer_backend vllm --vllm_enforce_eager"
            find_and_kill_process(kill_cmd)
        else:
            generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                --stage sft \
                --do_predict \
                --model_name_or_path {dpo_full_path} \
                --dataset {testset} \
                --dataset_dir {args.dataset_dir} \
                --template {args.template} \
                --finetuning_type {args.finetuning_type} \
                --output_dir {dpo_full_path} \
                --overwrite_cache \
                --overwrite_output_dir \
                --cutoff_len {args.cutoff_len} \
                --per_device_eval_batch_size {args.per_device_eval_batch_size} \
                --predict_with_generate \
                --report_to none\
                --fp16
            """

            run_cli_command(generate_text_command)
            jsonl_to_json(f"{dpo_full_path}/generated_predictions.jsonl", f"{args.dataset_dir}/generated_predictions_{testset}.json")
            
        # Add new dataset info to datset_info.json to run predict reward model
        add_new_dataset_info(args.data_info_path, dataset_name_generated, f"generated_predictions_{testset}.json")
        # -------------

        predict_rw_score_path = f"{oracle_adapter_path}/Iter_{iter}"
        if args.use_accelerate:
            if args.type_of_oracle == 'lora':
                inference_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
                --config_file examples/accelerate/single_config.yaml \
                src/train_bash.py \
                --stage rm \
                --do_predict \
                --flash_attn {args.flash_attn}\
                --model_name_or_path {args.model_name_or_path} \
                --adapter_name_or_path {oracle_adapter_path}\
                --dataset_dir {args.dataset_dir} \
                --dataset {dataset_name_generated} \
                --template {args.template} \
                --output_dir {oracle_adapter_path} \
                --per_device_eval_batch_size {args.per_device_eval_batch_size} \
                --fp16
                """

                print(f"Inference Oracle model ............................")
                run_cli_command(inference_oracle_command)
                # delete_item_dataset_info(args.data_info_path, dataset_name_generated) # clean dataset_info
                # -------------------------
                # Get accuracy        
                accuracy = calculate_accuracy(f"{predict_rw_score_path}/generated_predictions.jsonl")
                
                active_accuracy.append({
                    "iter": iter,
                    "acc": accuracy 
                })
                print("Active accuracy:", active_accuracy)
                save_eval_metric_path = f"{oracle_adapter_path}/active_acc_{iter}.json"
                save_eval_metric(save_eval_metric_path, accuracy, iter)
            else:
                inference_oracle_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
                    --stage oracle \
                    --do_eval \
                    --flash_attn {args.flash_attn}\
                    --model_name_or_path {args.model_name_or_path} \
                    --dataset_dir {args.dataset_dir} \
                    --dataset {dataset_name_generated} \
                    --cutoff_len {args.cutoff_len}\
                    --template {args.template} \
                    --output_dir {eval_metric_dir} \
                    --per_device_eval_batch_size {args.per_device_eval_batch_size} \
                    --is_compute_emb True\
                    --vhead_oracle_path {oracle_adapter_path}\
                    --active_iter {iter}\
                    --fp16\
                    --report_to none
                    """
        
                print(f"Inference Oracle model ............................")
                run_cli_command(inference_oracle_command)

            print("=========================================================")
        
    print("DONE!!!")

    # if not os.path.exists(eval_metric_dir):
    #     os.makedirs(eval_metric_dir)

    # json_filename = os.path.join(eval_metric_dir, "active_accuracy.json")
    # with open(json_filename, 'w') as json_file:
    #     json.dump(active_accuracy, json_file)
    # print(f"Results saved to {json_filename}")
    
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
    parser.add_argument("--is_using_vllm", action="store_true", help="Using vLLM to run 70B model")
    parser.add_argument("--flash_attn", action="store_true", help="Using Flash attention")
    parser.add_argument("--is_retrain_oracle", action="store_true", help="retrain oracle model if using custom trainer for oracle")
    parser.add_argument("--type_of_oracle", type=str, default="lora", help="Type of training Oracle: lora or v_head")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)  