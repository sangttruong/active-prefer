from datasets import load_dataset, Dataset
import pandas as pd
import json

def convert_multiple_choice_to_prompt(dataset, json_file_path):
    new_samples = []

    for question_id, example in enumerate(dataset):
        chosen = example["chosen"]
        rejected = example["rejected"]

        assist_idx = rejected.rfind("\n\nAssistant: ")
        r_reject = rejected[assist_idx+13:].strip()
        assist_idx = chosen.rfind("\n\nAssistant: ")
        r_accept = chosen[assist_idx+13:].strip()

        human_idx = chosen.rfind("\n\nHuman: ")
        query = chosen[human_idx+9:assist_idx].strip()
        prompt = chosen[:human_idx]
        history = []

        while prompt.rfind("\n\nAssistant: ") != -1:
            assist_idx = prompt.rfind("\n\nAssistant: ")
            human_idx = prompt.rfind("\n\nHuman: ")
            if human_idx != -1:
                old_query = prompt[human_idx+9:assist_idx].strip()
                old_resp = prompt[assist_idx+13:].strip()
                history.insert(0, (old_query, old_resp))
            else:
                break
            prompt = prompt[:human_idx]
        
        correct_choice_index = 0 
        correct_choice_text =  [r_accept, r_reject]

        for i, choice_text in enumerate(correct_choice_text):
            if i != correct_choice_index:
                new_samples.append({
                    "id": f"{question_id}",
                    "instruction": query,
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    splits = ['train', 'test']
    for split in splits:
        if args.sanity_check == 'True':
            dataset = load_dataset("Anthropic/hh-rlhf", split=f"{split}[:100]")
        else:
            dataset = load_dataset("Anthropic/hh-rlhf", split=f"{split}") 
        output_dataset_path = f'data/hellaswag_{split}.json'
        convert_multiple_choice_to_prompt(dataset, output_dataset_path)


    
