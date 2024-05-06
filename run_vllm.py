import argparse
import pandas as pd
import json
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

def get_prompt(dataset, dataset_dir = 'data'):
    dataset.replace(".json", "")
    dataset_path = f"{dataset_dir}/{dataset}.json"

    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    breakpoint()
    propmts = [x for x in data.valye]

    # TODO: only get instruction in data 

    return  [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

def run_infer(model_name_or_path, dataset_name, output_dir = 'save'):
    prompts = get_prompt(dataset_name)
    df = generate_texts(prompts, model_name_or_path)

    # Save DataFrame to a CSV file
    model_name = model_name_or_path.split('/')[-1]
    csv_path = f"{output_dir}/{model_name}/{dataset_name}/generated_texts.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated texts saved to {csv_path}")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--dataset_name", type=str, default="reward_bench_train", help="Test")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Test")
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_infer(args.model_name_or_path, args.dataset_name)
    
if __name__ == "__main__":
    main()
