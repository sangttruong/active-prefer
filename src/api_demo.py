# import os

# import uvicorn

# from llmtuner import ChatModel, create_app


# def main():
#     chat_model = ChatModel()
#     app = create_app(chat_model)
#     print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8005)))
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)), workers=1)


# if __name__ == "__main__":
#     main()


import os
from copy import deepcopy
import json
import argparse

from tqdm import tqdm

import uvicorn
from llmtuner import ChatModel, create_app

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="API Demo Arguments")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--testset", type=str, default="reward_bench_test", help="testset for generations")
    parser.add_argument("--template", type=str, required=True, help="Template to use")
    parser.add_argument("--infer_backend", type=str, required=True, help="Inference backend")
    parser.add_argument("--vllm_enforce_eager", action="store_true", help="Enforce eager execution for VLLM")
    parser.add_argument("--gpu_ids", type=str, default="3,4", help="Enforce eager execution for VLLM")
    parser.add_argument("--api_port", type=int, default=8005, help="Port for the API server")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["API_PORT"] = str(args.api_port)

    infer_args = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "infer_backend": args.infer_backend,
        "vllm_enforce_eager": args.vllm_enforce_eager
    }

    chat_model = ChatModel(infer_args)
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(args.api_port))
    uvicorn.run(app, host="0.0.0.0", port=args.api_port, workers=1)

    ### Begin inference DPO model
    client = OpenAI(
        base_url=f"http://localhost:{args.api_port}/v1",
        api_key="token-abc123",
    )


    testset_path = f"{args.dataset_dir}/{args.testset}.json" 
    with open(testset_path, 'r') as json_file:
        test_data = json.load(json_file)

    predictions = []
    for sample in tqdm(test_data):
        pred_sample = deepcopy(sample)

        completion = client.chat.completions.create(
            model=args.model_name_or_path,
            messages=[
                {"role": "user", "content": pred_sample['instruction']}
            ]
        )
        pred = completion.choices[0].message.content
        pred_sample['output'][0] = pred
        
        predictions.append(pred_sample)


    # Save result at "args.dataset_dir}/generated_predictions.json" 
    output_file_path = os.path.join(args.dataset_dir, "generated_predictions.json")
    # Save the predictions to the JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(predictions, output_file)
    print(f"Predictions saved to: {output_file_path}")

if __name__ == "__main__":
    main()
