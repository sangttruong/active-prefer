import argparse
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    breakpoint()
    return data

def main():
    parser = argparse.ArgumentParser(description='Read JSON Lines file')
    parser.add_argument('--file_path', type=str, help='Path to the JSON Lines file')
    args = parser.parse_args()

    jsonl_data = read_jsonl(args.file_path)
    print(jsonl_data)

if __name__ == "__main__":
    main()
