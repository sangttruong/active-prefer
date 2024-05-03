from datasets import load_dataset, Dataset
import pandas as pd
import json
import copy
import os

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for question_id, example in enumerate(dataset):
        correct_choice_index = 0 # 
        correct_choice_text = example["mc1_targets"]["choices"]
        correct_choice_label = example["mc1_targets"]["labels"]

        for i, (choice_text, choice_label) in enumerate(zip(correct_choice_text, correct_choice_label)):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"question_{question_id}",
                    "instruction": example["question"],
                    'input': "",
                    "output": [correct_choice_text[correct_choice_index], choice_text],
                    "choice_label": [correct_choice_index, i],
                })
    directory = os.path.dirname(json_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

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

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=str, default="False", help="Test")
    parser.add_argument("--model_name", type=str, default="llama2", help="Test")
    parser.add_argument("--dataset_info_path", type=str, default="data/dataset_info.json", help="Test")
    parser.add_argument("--method", type=str, default="random", help="Test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.sanity_check == 'True':
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation[:100]")
    else:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    dataset = dataset.train_test_split(test_size=0.2)
    splits = ['train', 'test']
    for split in splits:
        name = f"truthful_qa_{split}_{args.model_name}_{args.method}{'_check' if args.sanity_check == 'True' else ''}"
        output_dataset_path = f'data/{name}.json'
        convert_multiple_choice_to_prompt(dataset[split], output_dataset_path)
        add_new_dataset_info(args.dataset_info_path, name, f"{name}.json")
        print(f"{split}: {len(dataset)}")
    
