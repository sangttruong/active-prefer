import os

import uvicorn

from llmtuner import ChatModel, create_app




def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8005)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)), workers=1)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--testset", type=str, default="reward_bench_test", help="Test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    breakpoint()
    main()
