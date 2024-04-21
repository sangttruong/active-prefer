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
import argparse
import uvicorn
from llmtuner import ChatModel, create_app

def parse_args():
    parser = argparse.ArgumentParser(description="API Demo Arguments")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--template", type=str, required=True, help="Template to use")
    parser.add_argument("--infer_backend", type=str, required=True, help="Inference backend")
    parser.add_argument("--vllm_enforce_eager", action="store_true", help="Enforce eager execution for VLLM")
    parser.add_argument("--gpu_ids", type=str, default="3,4", help="Enforce eager execution for VLLM")
    parser.add_argument("--api_port", type=int, default=8005, help="Port for the API server")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["API_PORT"] = str(args.api_port)

    chat_model = ChatModel(args.model_name_or_path, args.template, args.infer_backend, args.vllm_enforce_eager)
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(args.api_port))
    uvicorn.run(app, host="0.0.0.0", port=args.api_port, workers=1)

if __name__ == "__main__":
    main()
