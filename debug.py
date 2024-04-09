import argparse
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    breakpoint()
    return data

def jsonl_to_json(jsonl_file_path, output_json_file_path):
    # Read JSONL file
    with open(jsonl_file_path, 'r') as jsonl_file:
        lines = jsonl_file.readlines()

    # Parse each line as JSON and store in a list
    json_data = [json.loads(line.strip()) for line in lines]

    # Write the list of JSON objects to a JSON file
    with open(output_json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Read JSON Lines file')
    parser.add_argument('--file_path', type=str, help='Path to the JSON Lines file')
    args = parser.parse_args()

    # jsonl_data = read_jsonl(args.file_path)
    jsonl_to_json(args.file_path, "data/gen.json")

if __name__ == "__main__":
    main()
