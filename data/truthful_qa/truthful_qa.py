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

if __name__ == "__main__":
    # Load the original dataset
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    output_dataset_path = '../truthful_qa.json'
    convert_multiple_choice_to_prompt(dataset, output_dataset_path)
    
