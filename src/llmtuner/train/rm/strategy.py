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

# from .workflow import CustomDataset, ValueHead, set_seed

from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead
from datasets import Dataset
from datasets import load_dataset


from safetensors import safe_open
from safetensors.torch import save_file, load_file

# import deepspeed
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


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, embeddings_feature, dataset, is_load=False):
        self.embeddings_feature = embeddings_feature # tuple
        self.dataset = dataset
        self.is_load = is_load

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        example = self.dataset[i]
        
        if self.is_load:
            return {"question_id": example['id'], # string 
                    "last_hidden_state_chosen": torch.tensor(self.embeddings_feature[f'arr_{i}'][0]), # tensor (ctx x 4096)
                    "last_hidden_state_rejected": torch.tensor(self.embeddings_feature[f'arr_{i}'][1]),  # tensor (ctx x 4096)
                    # 'chosen_ids': torch.tensor(example['chosen_ids']), # list ids
                    # 'rejected_ids': torch.tensor(example['rejected_ids']), # list ids
                    }
        else:
            return {"question_id": example['id'], # string 
                    "last_hidden_state_chosen": self.embeddings_feature[i][0].clone(), # tensor (ctx x 4096)
                    "last_hidden_state_rejected": self.embeddings_feature[i][1].clone(),  # tensor (ctx x 4096)
                    # 'chosen_ids': torch.tensor(example['chosen_ids']), # list ids
                    # 'rejected_ids': torch.tensor(example['rejected_ids']), # list ids
                    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        # self.v_head = ValueHead(self.base_model.config)

        self.dataset = self.data_args.dataset

    def query(self, n):
        # Select instances from the pool for labeling
        pass

    def _train_vhead(self, emb_dataset, val_size= 0.1, random_state = 0):
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

        # Save the model to a file
        output_path = f"{self.training_args.output_dir}/vhead_selector.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        return accuracy

    def train(self, is_compute_emb, val_size = 0.1):
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)

        metrics = []
        print(f"Trainig selector  ...................")
        val_acc = self._train_vhead(emb_dataset, val_size)

        metrics.append({
            "Accuracy": val_acc,
        })
        print(metrics)

        output_file = f"{self.training_args.output_dir}/seletor_model.json"
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

    def predict(self, question_ids=None):
        # Predict labels for given data
        accelerator = Accelerator()
        device = accelerator.device
        
        last_hidden_states, is_load = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset, is_load) 

        self.v_head.eval()
        predictions = []
        with torch.no_grad():
            for idx in tqdm(range(len(train_dataset))):
                example = train_dataset[idx]
                last_hidden_state_chosen = example['last_hidden_state_chosen'][-1].to(device)
                last_hidden_state_rejected = example['last_hidden_state_rejected'][-1].to(device)

                chosen_rewards = self.v_head(last_hidden_state_chosen)
                rejected_rewards = self.v_head(last_hidden_state_rejected) 

                pred = {"question_id": example['question_id'],
                        "chosen_rewards": chosen_rewards,
                        "rejected_rewards": rejected_rewards
                }

                predictions.append(pred)
    
        return predictions

    def predict_prob(self, model):
        # Predict probabilities for given data
        emb_dataset = self.get_training_dataset(True)

        chosen_emb = np.array(emb_dataset['chosen'])
        rejected_emb = np.array(emb_dataset['rejected'])

        breakpoint()
        
        chosen_scores = model.predict_proba(chosen_emb) # [n_samples, n_classes]
        rejected_scores = model.predict_proba(rejected_emb) # [n_samples, n_classes]
        
        # Get max values along axis -1 for chosen and rejected scores
        max_chosen_scores = np.max(chosen_scores, axis=-1)
        max_rejected_scores = np.max(rejected_scores, axis=-1)

        # Concatenate max values of chosen and rejected scores
        max_scores_concat = np.concatenate((max_chosen_scores, max_rejected_scores), axis=-1)

        # Compute softmax over concatenated max scores
        softmax_scores = F.softmax(torch.tensor(max_scores_concat), dim=0).numpy()


        # Assuming you want to keep chosen and rejected probabilities separate
        pred = {
            "question_id": emb_dataset['question_id'],
            "chosen_rewards": softmax_scores[:, 0],
            "rejected_rewards": softmax_scores[:, 1],
        }
        return pred
        
    def predict_prob_dropout(self, n_drop):
        # Predict probabilities using dropout for uncertainty estimation

        accelerator = Accelerator()
        device = accelerator.device
        
        last_hidden_states, is_load = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset, is_load) 

        self.v_head.eval()
        predictions = {}
        with torch.no_grad():
            for _ in range(n_drop):
                for idx in range(len(train_dataset)):
                    example = train_dataset[idx]
                    last_hidden_state_chosen = example['last_hidden_state_chosen'][-1].to(device)
                    last_hidden_state_rejected = example['last_hidden_state_rejected'][-1].to(device)

                    chosen_rewards = self.v_head(last_hidden_state_chosen)
                    rejected_rewards = self.v_head(last_hidden_state_rejected) 
                    rewards_concat = torch.tensor([chosen_rewards, rejected_rewards], device=device)
                    
                    probs = F.softmax(rewards_concat, dim=0)

                    question_id = example['question_id']
                    if question_id in predictions:
                        predictions[question_id]['chosen_rewards'] += probs[0].item()
                        predictions[question_id]['rejected_rewards'] += probs[1].item()
                    else: 
                        predictions[question_id] = {
                            "chosen_rewards": probs[0].item(),  
                            "rejected_rewards": probs[1].item() 
                        }
        
        for question_id in predictions.keys():
            predictions[question_id]["chosen_rewards"] /= n_drop
            predictions[question_id]["rejected_rewards"] /= n_drop

        return predictions

    def predict_prob_dropout_split(self, X, Y, n_drop):
        # Predict probabilities using dropout but return individual dropout iterations
        pass

    def get_embedding(self, is_override = False, hf_emb_path = None):
        # Get embeddings from the penultimate layer of the network
        filename = f"{self.training_args.output_dir}/last_hidden_states"
        # Check if the file exists

        if hf_emb_path is not None:
            train_df = load_dataset(hf_emb_path)
        if is_override == False and os.path.isfile(f"{filename}.pkl"):
            train_ds = pickle.load(open(f"{filename}.pkl", 'rb'))
            train_df = Dataset.from_dict(train_ds)
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

            # Initialize the DeepSpeed-Inference engine
            # ds_engine = deepspeed.init_inference(self.base_model.model)

            with torch.no_grad():
                for batch in tqdm(dataloader):
                    emb = self.base_model.model(**batch).last_hidden_state # (bz, ctx, 4096)
                    # emb = ds_engine(**batch)

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
            train_df = Dataset.from_dict(vector_output)


            # Free GPU memory consumed by model parameters
            # ds_engine.empty_partition_cache()

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