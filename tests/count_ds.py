import json

def count_len_dataset(prediction_path):
    with open(prediction_path, 'r') as f:
        data = json.load(f)
        ids = set(item['id'] for item in data)
    
    return len(ids)

if __name__ == "__main__":
    ds_path = '../data/arc_sample.json'
    count = count_len_dataset(ds_path)
    print(count)
