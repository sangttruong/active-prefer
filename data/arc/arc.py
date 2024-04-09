from datasets import load_dataset, Dataset
import pandas as pd
import json

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for example in dataset:
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


    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=str, default="False", help="Test")
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
        output_dataset_path = f'data/arc_challenge_{split}.json'
        convert_multiple_choice_to_prompt(dataset, output_dataset_path)
