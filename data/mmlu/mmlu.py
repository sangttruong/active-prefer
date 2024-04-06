from datasets import load_dataset, Dataset
import pandas as pd
import json

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for question_id, example in enumerate(dataset):
        question = example["question"]
        correct_choice_index = example['answer'] 
        correct_choice_text = example["choices"]

        for i, choice_text in enumerate(correct_choice_text):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"{example['subject']}_{question_id}",
                    "instruction": question,
                    'input': "",
                    "output": [correct_choice_text[correct_choice_index], choice_text],
                    "choice_label": [correct_choice_index, i],
                })

    # Write to JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(new_samples, json_file, indent=4)

if __name__ == "__main__":
    # Load the original dataset
    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    
    output_dataset_path = '../mmlu.json'
    convert_multiple_choice_to_prompt(dataset, output_dataset_path)
    
