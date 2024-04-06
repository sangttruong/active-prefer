from datasets import load_dataset, Dataset
import pandas as pd
import json

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for question_id, example in enumerate(dataset):
        instruction = example['ctx']
        correct_choice_index = int(example['label'])
        correct_choice_text = example["endings"]

        for i, choice_text in enumerate(correct_choice_text):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"question_{question_id}",
                    "instruction": instruction,
                    'input': "",
                    "output": [correct_choice_text[correct_choice_index], choice_text],
                    "choice_label": [correct_choice_index, i],
                })

    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

if __name__ == "__main__":
    # Load the original dataset
    dataset = load_dataset("Rowan/hellaswag", split="train[:20]")
    
    output_dataset_path = '../hellaswag.json'
    convert_multiple_choice_to_prompt(dataset, output_dataset_path)
    
