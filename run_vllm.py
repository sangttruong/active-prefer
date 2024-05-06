from vllm import LLM, SamplingParams
import pandas as pd

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
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

# create df
results = {"Prompt": prompts}

K = 5
for k in range(K):
    # Infer
    outputs = llm.generate(prompts, sampling_params)
    
    results[f"Generated_Text_{k}"] = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results[f"Generated_Text_{k}"].append(generated_text)

df = pd.DataFrame(results)
model_name = model_name_or_path.split('/')[-1]
df.to_csv(f"saves/generated_text_{model_name}.csv", index=False)

        
