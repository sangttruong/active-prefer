from datasets import load_dataset, Dataset
import pandas as pd
import json

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

    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=bool, default=False, help="Test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.sanity_check:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation[:100]")
    else:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    dataset = dataset.train_test_split(test_size=0.2)
    splits = ['train', 'test']
    for split in splits:
        output_dataset_path = f'data/truthful_qa_{split}.json'
        convert_multiple_choice_to_prompt(dataset[split], output_dataset_path)
    
