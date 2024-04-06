import json

def select_top_percent_questions_from_file(file_path, percent):
    question_scores = {}

    # Read data from JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            question_id = entry["question"]
            chosen_score = entry["chosen"]
            rejected_scores = entry["rejected"]

            # check exits
            if question_id in question_scores:
                question_scores[question_id].append(rejected_scores)
            else:
                question_scores[question_id] = [chosen_score]

    # Calculate the average of averages for each question
    from scipy.stats import entropy
    
    entropy_vals = {}
    for question_id, scores in question_scores.items():
        entropy_value = entropy(scores, base=2)
        entropy_vals[question_id] = entropy_value

    # Sort the questions based on their average scores
    sorted_entropy = sorted(entropy_vals.items(), key=lambda x: x[1], reverse=True)

    # Calculate the number of questions to select based on the percentage
    num_questions = len(sorted_entropy)
    num_questions_to_select = int(num_questions * percent)

    # Get the top questions based on the specified percentage
    top_percent_questions = [question[0] for question in sorted_entropy[:num_questions_to_select]]

    return top_percent_questions


def select_entries_by_ids(data_path, id_list, output_file):
    selected_entries = []

    # Read data from dataset.json
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = []
    for entry in data:
        if entry["id"] in id_list:
            selected_entries.append(entry)
        else:
            remaining_entries.append(entry)

    # Write remaining entries back to the original file
    with open(data_path, 'w') as file:
        json.dump(remaining_entries, file, indent=4)

    # Write selected entries to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(selected_entries, outfile, indent=4)

def delete_selected_info(data_info_path, pattern="iter"):
    # Read data from dataset.json
    with open(data_info_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = {}
    for name, v in data.items():
        if pattern not in name:
            remaining_entries[name] = v

    # Save the remaining data info
    with open(data_info_path, 'w') as outfile:
        json.dump(remaining_entries, outfile, indent=4)

    print("New data info with selected entries removed has been saved.")

def count_len_datset(prediction_path):
    # Open the file and read its lines, then get the count of lines
    with open(prediction_path, 'r') as file:
        data_count = len(file.readlines())
    
    # Return the count of data
    return data_count

import random

# Mock function to simulate reading inference results
def read_inference_results(prediction_path):
    # This is a mock function, replace it with your actual implementation
    # For testing, let's assume it returns a dictionary with some sample scores
    return {
        "question1": 0.8,
        "question2": 0.6,
        "question3": 0.9,
        "question4": 0.5,
        "question5": 0.7
    }

# Function to test
def select_by_random(prediction_path, percent_or_num):
    question_scores = {}
    question_scores = read_inference_results(prediction_path)
    all_question_ids = list(set(question_scores.keys()))
    num_questions = len(all_question_ids)
    if percent_or_num < 1:
        num_questions_to_select = int(num_questions * percent_or_num)
    else:
        num_questions_to_select = min(num_questions, int(percent_or_num))
    selected_questions = random.sample(all_question_ids, num_questions_to_select)
    return selected_questions




def main():
    # # Example usage:
    # prediction_path = "data/prediction.jsonl"
    # percentage = 0.1
    # selected_questions = select_top_percent_questions_from_file(prediction_path, percentage)  # Select top 10% questions
    # print("Top 10% questions based on max entropy:", selected_questions)

    # # Example usage:
    # data_path = "data/sample.json"
    # output_file = "data/selected_entries.json"  # Name of the output file
    # select_entries_by_ids(data_path, selected_questions, output_file)
    # print("Selected entries have been stored in", output_file)


    # # Read data from dataset.json
    # data_info_path = 'data/dataset_info.json'
    # dataset_name = "arc_sample"
    # with open(data_info_path, 'r') as file:
    #     data_info = json.load(file)
        
    #     if dataset_name in data_info:
    #         # append new info to data_infor and store result in json file
    #         new_name = dataset_name + "_iter_0"
    #         data_info[new_name] = data_info[dataset_name]
    #         data_info[new_name]["file_name"] = "selected_entries.json"

    #         with open(data_info_path, 'w') as outfile:
    #             json.dump(data_info, outfile, indent=4)

    #         print("Updated dataset info has been stored in", data_info_path)

    # Write the sample data info to a temporary file
    # data_info_path = 'data/dataset_info.json'
    # delete_selected_info(data_info_path)

    # ========================================
    # output_file = "data/selected_entries.json"  # Name of the output file
    # print(read_and_return_len(output_file))
    # pass

    # ===================================
    # Test the function
    prediction_path = "data/prediction.jsonl"
    percent_or_num = 0.2  # Select 50% of questions randomly
    selected_questions = select_by_random(prediction_path, percent_or_num)
    print("Selected questions:", selected_questions)

    
if __name__ == "__main__":
    main()


