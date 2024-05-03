from datasets import load_dataset, Dataset
import pandas as pd
import json
import copy
import os
from tqdm import tqdm

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for example in tqdm(dataset):
        correct_choice_index = example["choices"]["label"].index(example["answerKey"])
        correct_choice_text = example["choices"]["text"][correct_choice_index]
        correct_choice_label = example["choices"]["label"][correct_choice_index]

        for i, (choice_text, choice_label) in enumerate(zip(example["choices"]["text"], example["choices"]["label"])):
            if i != correct_choice_index:
                new_samples.append({
                    "id": example["id"],
                    "instruction": example["question"],
                    'input': "",
                    "output": [correct_choice_text, choice_text],
                    "choice_label": [correct_choice_label, choice_label],
                })

    directory = os.path.dirname(json_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

def add_new_dataset_info(dataset_info_path, name, path):
    # Ensure directory exists, create it if it doesn't
    directory = os.path.dirname(dataset_info_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=str, default="False", help="Test")
    parser.add_argument("--model_name", type=str, default="llama2", help="Test")
    parser.add_argument("--dataset_info_path", type=str, default="data/dataset_info.json", help="Test")
    parser.add_argument("--method", type=str, default="random", help="Test")

    return parser.parse_args()


if __name__ == "__main__":
    # Load the original dataset
    args = parse_arguments()

    splits = ['train', 'test']
    for split in splits:
        
        if args.sanity_check == 'True':
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f'{split}[:50]')
        else:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f'{split}')
        
        name = f"arc_challenge_{split}_{args.model_name}_{args.method}{'_check' if args.sanity_check == 'True' else ''}"
        output_dataset_path = f'data/{name}.json'
        convert_multiple_choice_to_prompt(dataset, output_dataset_path)
        add_new_dataset_info(args.dataset_info_path, name, f"{name}.json")

        print(f"{split}: {len(dataset)}")