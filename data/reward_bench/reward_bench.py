from datasets import load_dataset, Dataset
import pandas as pd
import json
import copy


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

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for question_id, example in enumerate(dataset):
        question = example["prompt"]
        correct_choice_index = 0 
        correct_choice_text = [example["chosen"], example["rejected"]]

        for i, choice_text in enumerate(correct_choice_text):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"{example['id']}",
                    "instruction": question,
                    'input': "",
                    "output": [correct_choice_text[correct_choice_index], choice_text],
                    "choice_label": [correct_choice_index, i],
                })

    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

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
        dataset = load_dataset("allenai/reward-bench", split="train[:100]")
    else:
        dataset = load_dataset("allenai/reward-bench", split="train")
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
    splits = ['train', 'test']
    for split in splits:
        print(f"{split}: {len(dataset[split])}")
        if args.sanity_check == "True":
            name = f"reward_bench_{split}_{args.model_name}_{args.method}_check"
            output_dataset_path = f'data/{name}.json'
            convert_multiple_choice_to_prompt(dataset[split], output_dataset_path)
            add_new_dataset_info(args.dataset_info_path, name, f"{name}.json")
        else:
            name = f"reward_bench_{split}_{args.model_name}_{args.method}"
            output_dataset_path = f'data/{name}.json'
            convert_multiple_choice_to_prompt(dataset[split], output_dataset_path)
            add_new_dataset_info(args.dataset_info_path, name, f"{name}.json")
    
