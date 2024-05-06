import pandas as pd
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
model_name_or_path = "meta-llama/Llama-2-7b-hf"
llm = LLM(model=model_name_or_path)

# Initialize a dictionary to store the results.
results = {"Prompt": []}

# Generate texts from the prompts.
K = 5
for k in range(1, K + 1):
    results[f"Generated_Text_{k}"] = []

for prompt in prompts:
    results["Prompt"].append(prompt)
    outputs = llm.generate([prompt] * K, sampling_params)
    for k, output in enumerate(outputs, start=1):
        generated_text = output.outputs[0].text
        results[f"Generated_Text_{k}"].append(generated_text)

# Convert the results dictionary to a DataFrame.
df = pd.DataFrame(results)

# Display the DataFrame.
print(df)
