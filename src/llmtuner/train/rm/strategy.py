import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy

from torch.distributions.categorical import Categorical


from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import gc
import pickle
import os 
import numpy as np
import random
import json
import copy
from tqdm import tqdm

from ...data import get_dataset, split_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .collator import PairwiseDataCollatorWithPadding
from .metric import compute_accuracy
from .trainer import PairwiseTrainer, OracleTrainer
from ..utils import load_valuehead_params 


from datasets import Dataset
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4)
    jsonFile = open(name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def save_to_pkl(data, name):
    pklFile = open(name, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

class LLMStrategy:
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args

        self.tokenizer = load_tokenizer(model_args)
        if "mistral" in self.model_args.model_name_or_path.lower():  
            self.tokenizer.padding_side  = 'left'
            

        self.pool_dataset = get_dataset(self.tokenizer, model_args, data_args, training_args, stage="rm")
        nearest_multiple = len(self.pool_dataset) // 8 * 8
        self.pool_dataset = self.pool_dataset.select(list(range(nearest_multiple)))
        
        self.base_model = load_model(self.tokenizer, model_args, finetuning_args, False, add_valuehead=False)
        self.data_collator = PairwiseDataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        self.callbacks = callbacks

        # Replace lm_head with identity
        # if hasattr(self.base_model, "lm_head"):
        #     self.base_model.lm_head = torch.nn.Identity()

        # Update arguments
        training_args.remove_unused_columns = False  

        # Initialize our Trainer
        self.trainer = PairwiseTrainer(
            model=self.base_model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=callbacks + [FixValueHeadModelCallback()],
            compute_metrics=compute_accuracy,
            **split_dataset(self.pool_dataset, self.data_args, self.training_args),
        )

        # Model

        self.dataset = self.data_args.dataset

    def query(self, n):
        # Select instances from the pool for labeling
        pass

    def train_vhead(self, emb_dataset, val_size= 0.1, random_state = 0):
        X1 = np.array(emb_dataset['chosen'])
        X2 = np.array(emb_dataset['rejected'])        
        X = X1 - X2 # chosen - rejected
        y = np.ones(len(X)) # chosen = 1

        chosen_index = np.random.choice(len(X), size=int(0.5 * len(X)), replace=False)
        for idx in chosen_index:    
            X[idx] = -X[idx]
            y[idx] = 0 # chosen label

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=random_state)

        # Fitting the logistic regression model
        model = LogisticRegression(random_state=random_state, max_iter=500)
        model.fit(X_train, y_train)

        # Once the model is trained, you can evaluate its performance on the test set
        accuracy = model.score(X_test, y_test)
        print("Accuracy on test set:", accuracy)

        return accuracy, model

    def eval_vhead(self, emb_testset, model_path):
        # Load the model from a file
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
        X = np.array(emb_testset['chosen'])
        y = np.ones(len(X)) # chosen = 1

        # Once the model is trained, you can evaluate its performance on the test set
        accuracy = model.score(X, y)
        print("Accuracy on test set:", accuracy)
        return accuracy
    
    def predict_prob(self, emb_dataset, model_path):
        # Predict probabilities for given data
        # emb_dataset = self.get_training_dataset(is_compute_emb)

        # Load the model from a file
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
        chosen_emb = np.array(emb_dataset['chosen'])
        rejected_emb = np.array(emb_dataset['rejected'])
        
        chosen_scores_prob = model.predict_log_proba(chosen_emb) # [n_samples, n_classes]
        rejected_scores_prob = model.predict_log_proba(rejected_emb) # [n_samples, n_classes]
        
        # Get max values along axis -1 for chosen and rejected scores
        chosen_scores = chosen_scores_prob[:, 1].reshape(-1, 1)
        rejected_scores = rejected_scores_prob[:, 0].reshape(-1, 1)

        # Concatenate max values of chosen and rejected scores
        max_scores_concat = np.concatenate((chosen_scores, rejected_scores), axis=-1)

        # Compute softmax over concatenated max scores
        softmax_scores = F.softmax(torch.tensor(max_scores_concat), dim=-1).numpy()

        # Assuming you want to keep chosen and rejected probabilities separate
        pred = {
            "question_id": emb_dataset['question_id'],
            "chosen_rewards": softmax_scores[:, 0],
            "rejected_rewards": softmax_scores[:, 1],
        }
        return pred

    def train(self, nEns, is_compute_emb, val_size = 0.1):
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)

        metrics = []
        for m in tqdm(range(nEns)):                        
            print(f"Trainig selector {m} ...................")
            model_path = f"{self.training_args.output_dir}/vhead_{m}.pkl"
            val_acc = self.train_vhead(emb_dataset, model_path, val_size, random_state=m)

            metrics.append({
                "model_id": m,
                "Accuracy": val_acc,
            })
            print(metrics)

        output_file = f"{self.training_args.output_dir}/vhead.json"
        with open(output_file, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
    
    def getNet(self, params):
        # Construct and return a model given parameters
        i = 0
        model = deepcopy(self.v_head).cuda()
        for p in model.parameters():
            L = len(p.flatten())
            param = params[i:(i + L)]
            p.data = param.view(p.size())
            i += L
        return model
    
    def sampleNet(self, weights, iterates):
        # Sample a model from the weights and parameter updates
        nEns = len(weights)
        k = len(iterates[0])
        i = np.random.randint(nEns)
        z = torch.randn(k, 1)
        weightSample = weights[i].view(-1) - torch.mm(iterates[i].t(), z).view(-1) / np.sqrt(k)
        sampleNet = self.getNet(weightSample).cuda()
        return sampleNet

    def getPosterior(self, weights, iterates, X, Y, nSamps=50):
        # Estimate the posterior distribution of class labels given model uncertainty

        model = self.sampleNet(weights, iterates)
        output = self.predict_prob(X, Y, model=model) / nSamps
        print(' ', flush=True)
        ce = nn.CrossEntropyLoss()
        print('sampling models', flush=True)
        for i in range(nSamps - 1):
            model = self.sampleNet(weights, iterates)
            output = output + self.predict_prob(X, Y, model=model) / nSamps
            print(i+2, torch.sum(torch.argmax(output, 1) == Y).item() / len(Y), flush=True)
        return output.numpy()

    

    def get_embedding(self, is_override = False, hf_emb_path = None):
        # Get embeddings from the penultimate layer of the network
        filename = f"{self.training_args.output_dir}/last_hidden_states"
        # Check if the file exists

        if hf_emb_path is not None:
            train_df = load_dataset(hf_emb_path)
        elif is_override == False and os.path.isfile(f"{filename}.pkl"):
            train_df = pickle.load(open(f"{filename}.pkl", 'rb'))
            print(f"Loaded data from {filename}")
        else:
            self.base_model.eval()
            # ------------------------------------------------------
            print("Begin complute emb..........")
            dataloader = self.trainer.get_test_dataloader(self.pool_dataset)
            vector_output = {
                "question_id": self.pool_dataset['id'],
                "chosen": [],
                "rejected": []
            }

            with torch.no_grad():
                for batch in tqdm(dataloader):
                    emb = self.base_model.model(**batch).last_hidden_state # (bz, ctx, 4096)
                    bz, ctx , _ = emb.shape
                    emb = emb.cpu()
                    # find item != 2, set 1
                    mask = torch.zeros((bz, ctx))
                    last_token_index  = ((batch['input_ids'] != self.tokenizer.pad_token_id).sum(-1) - 1).tolist() 
                    for row, col in enumerate(last_token_index):
                        mask[row, col] = 1
                    # Mul
                    emb_mul = emb * mask.unsqueeze(-1)
                    # Sum
                    last_token_emb = emb_mul.sum(1)
                    chosen_emb = [np.array(subarray) for subarray in last_token_emb[:bz//2]]
                    rejected = [np.array(subarray) for subarray in last_token_emb[bz//2:]]
                    vector_output["chosen"].extend(chosen_emb)
                    vector_output["rejected"].extend(rejected)  
                    
            # Stack 
            vector_output["chosen"] = np.stack(vector_output["chosen"], axis = 0)
            vector_output["rejected"] = np.stack(vector_output["rejected"], axis = 0)

            save_to_pkl(vector_output, f"{filename}.pkl")
            train_df = vector_output
        return train_df

    def get_training_dataset(self, is_override):
        train_dataset = self.get_embedding(is_override)
        return train_dataset
        
    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[]):
        # Calculate gradient embeddings for the data
        pass

    # fisher embedding for bait (assumes cross-entropy loss)
    def get_exp_grad_embedding(self, X, Y, probs=[], model=[]):
        # Calculate expected gradient embeddings for the data
        pass

    def update(self, question_ids, iteration = 0):
        selected_path = f"{self.data_args.dataset_dir}/selected_entries_{self.dataset}.json"  
        dataset_info_path = f"{self.data_args.dataset_dir}/dataset_info.json"
        dataset_path = f"{self.data_args.dataset_dir}/{self.dataset}.json"

        # Update data training
        select_entries_by_ids(dataset_path, question_ids, selected_path)

        with open(dataset_info_path, 'r') as file:
            data_info = json.load(file)
            
            if self.dataset in data_info:
                # append new info to data_infor and store result in json file
                new_data_info = f"{self.dataset}_iter_{iteration}"
                data_info[new_data_info] = copy.deepcopy(data_info[self.dataset])
                data_info[new_data_info]["file_name"] = f"selected_entries_{self.dataset}.json"

                with open(dataset_info_path, 'w') as outfile:
                    json.dump(data_info, outfile, indent=4)

                print(f"{new_data_info} updated in {dataset_info_path}")


def select_entries_by_ids(dataset_path, question_ids, seleted_path):
    selected_entries = []

    # Read data from dataset.json
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = []
    for entry in data:
        if entry["id"] in question_ids:
            selected_entries.append(entry)
        else:
            remaining_entries.append(entry)

    # Write remaining entries back to the original file
    with open(dataset_path, 'w') as file:
        json.dump(remaining_entries, file, indent=4)

    # Write selected entries to a new JSON file
    with open(seleted_path, 'w') as outfile:
        json.dump(selected_entries, outfile, indent=4)