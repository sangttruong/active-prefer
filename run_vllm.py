import argparse
import pandas as pd
import json
import os
from vllm import LLM, SamplingParams

def generate_texts(prompts, model_name_or_path, K=5, temperature=0.8, top_p=0.95):
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    # Create an LLM.
    llm = LLM(model=model_name_or_path)

    # Initialize a dictionary to store the results.
    results = {"Prompt": prompts}

    for k in range(K):
        # Infer
        outputs = llm.generate(prompts, sampling_params)

        results[f"Generated_Text_{k+1}"] = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results[f"Generated_Text_{k+1}"].append(generated_text)

    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def get_prompts(dataset, dataset_dir = 'data'):
    
    dataset_path = f"{dataset_dir}/{dataset}_train_Llama-2-7b-hf_random.json"

    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    propmts = [x['instruction'] for x in data]
    return propmts

def run_infer(model_name_or_path, dataset, output_dir = 'save'):
    prompts = get_prompts(dataset)
    df = generate_texts(prompts, model_name_or_path)

    # Save DataFrame to a CSV file
    model_name = model_name_or_path.split('/')[-1]
    csv_dir = os.path.join(output_dir, model_name, dataset)
    os.makedirs(csv_dir, exist_ok=True)  # Create directory if it doesn't exist
    csv_path = os.path.join(csv_dir, 'generated_texts.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated texts saved to {csv_path}")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--dataset", type=str, default="reward_bench_train", help="Test")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Test")
    parser.add_argument("--K", type=int, default=10, help="num iteration")

    return parser.parse_args()

def main():
    args = parse_arguments()
    run_infer(args.model_name_or_path, args.dataset)
    
if __name__ == "__main__":
    main()
