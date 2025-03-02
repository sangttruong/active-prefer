from datasets import load_dataset, Dataset
import pandas as pd
import json
import copy
import os

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    template = f"""Complete the sentence below by filling in the blank with an appropriate word: {{sentence}}\nAnswer: """
    new_samples = []

    for question_id, example in enumerate(dataset):
        instruction = example['sentence']
        correct_choice_index = int(example['answer']) - 1 # index start from 0
        correct_choice_text = [example["option1"], example["option2"]]

        for i, choice_text in enumerate(correct_choice_text):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"question_{question_id}",
                    "instruction": template.format(sentence = instruction),
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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Test")
    parser.add_argument("--dataset_info_path", type=str, default="data/dataset_info.json", help="Test")
    parser.add_argument("--method", type=str, default="random", help="Test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.sanity_check == "True":
        dataset = load_dataset("winogrande", "winogrande_debiased", split=f"train[:100]")
    else:
        dataset = load_dataset("winogrande", "winogrande_debiased", split=f"train")

    dataset = dataset.train_test_split(test_size=0.2)
    splits = ['train', 'test']
    for split in splits:
        model_name = args.model_name.split('/')[-1]
        name = f"winogrande_{split}_{model_name}_{args.method}{'_check' if args.sanity_check == 'True' else ''}"
        output_dataset_path = f'data/{name}.json'
        convert_multiple_choice_to_prompt(dataset[split], output_dataset_path)
        add_new_dataset_info(args.dataset_info_path, name, f"{name}.json")

        print(f"{split}: {len(dataset[split])}")



    
