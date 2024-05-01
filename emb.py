from dataclasses import dataclass, field
from typing import Optional


import torch
import json
import pickle
import pandas as pd
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
import pickle 
from datasets import Dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from huggingface_hub import login
login("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")


@dataclass
class ScriptArguments:
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="allenai/reward-bench", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    eval_split: Optional[str] = field(
        default="test", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})

    ##### EDIT HERE #####
    seq_length: Optional[int] = field(default=4096, metadata={"help": "Input sequence length"})
    # >>> GPT: 512; LLaMa-2: 4096

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})

    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})

    num_proc: Optional[int] = field(default=8, metadata={"help": "Number of processors used for preprocessing dataset"})

    saving_freq: Optional[int] = field(default=1000, metadata={"help": "Saving frequency of vector dataset"})
    #####################

def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4)
    jsonFile = open(name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def save_to_pkl(data, name):
    pklFile = open(name, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit,
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = {"":0}
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name
    )

    ## Fix GPT bug
    if 'gpt' in script_args.model_name:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(examples["prompt"] + chosen, truncation=True)
            tokenized_rejected = tokenizer(examples["prompt"] + rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples


    # Preprocess the dataset and filter out examples that are longer than script_args.max_length
    train_dataset = load_dataset(script_args.dataset_name, split="train")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=script_args.num_proc,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )

    if script_args.eval_split == "none":
        eval_dataset = None
    else:
        eval_dataset = load_dataset(script_args.dataset_name, split=script_args.eval_split)

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=script_args.num_proc,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
            and len(x["input_ids_rejected"]) <= script_args.seq_length
        )    



    vector_output = {
        "chosen": [],
        "rejected": []
    }
    TOTAL_TRAIN = len(train_dataset)
    print("Processing training set....")
    for idx, sample in tqdm(enumerate(train_dataset)):
        input_ids = torch.tensor([sample['input_ids_chosen']], device=model.device)
        input_mask = torch.tensor([sample['attention_mask_chosen']], device=model.device)
        with torch.no_grad():
            embeddings = model.model(input_ids=input_ids,
                                    attention_mask=input_mask).last_hidden_state[0][-1] # Get last token emb
        vector_output["chosen"].append(embeddings.cpu().numpy().tolist())


        input_ids = torch.tensor([sample['input_ids_rejected']], device=model.device)
        input_mask = torch.tensor([sample['attention_mask_rejected']], device=model.device)
        with torch.no_grad():
            embeddings = model.model(input_ids=input_ids,
                                    attention_mask=input_mask).last_hidden_state[0][-1] # Get last token emb
        vector_output["rejected"].append(embeddings.cpu().numpy().tolist())

        if (idx + 1) == TOTAL_TRAIN or (idx + 1) % script_args.saving_freq == 0:
            save_to_pkl(vector_output, "reward-bench-embedding-train.pkl")


    train_ds = pickle.load(open("reward-bench-embedding-train.pkl", 'rb'))
    train_df = Dataset.from_dict(train_ds)
    
    # Traing last layer
    X1 = np.array(train_df['chosen'])
    X2 = np.array(train_df['rejected'])        
    X = X1 - X2 # chosen - rejected
    y = np.ones(len(X)) # chosen = 1

    chosen_index = np.random.choice(len(X), size=int(0.5 * len(X)), replace=False)
    for idx in chosen_index:    
        X[idx] = -X[idx]
        y[idx] = 0 # chosen label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Fitting the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=5000)
    model.fit(X_train, y_train)

    # Once the model is trained, you can evaluate its performance on the test set
    accuracy = model.score(X_test, y_test)
    print("Accuracy on test set:", accuracy)

