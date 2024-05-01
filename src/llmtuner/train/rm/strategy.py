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
        self.v_head = ValueHead(self.base_model.config)

        self.dataset = self.data_args.dataset

    def query(self, n):
        # Select instances from the pool for labeling
        pass

    def _train_model(
        self,
        train_dataset, 
        cutoff_len, 
        pad_token_id, 
        optimizer_params, 
        create_scheduler, 
        num_epochs,
        model = None,
        sample_ids = None,
        seed = 42,
        save_path=None,  
    ):

        set_seed(seed)  # Set seed for reproducibility   
        
        accelerator = Accelerator()
        device = accelerator.device

        model = model.to(device) 

        optimizer_params.pop('params', None)     
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        
        num_epochs = int(num_epochs)
        num_training_steps_per_epoch = len(train_dataset) 
        num_training_steps = num_epochs * num_training_steps_per_epoch
        if sample_ids is None:
            sample_ids = list(range(len(train_dataset)))
        
        scheduler = create_scheduler(num_training_steps, optimizer = optimizer)

        model, optimizer, train_dataset = accelerator.prepare(model, optimizer, train_dataset)

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            for idx in sample_ids:
                example = train_dataset[idx]
                last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
                last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
                chosen_input_ids = example['chosen_ids']
                rejected_input_ids = example['rejected_ids']

                optimizer.zero_grad()
                chosen_rewards = model(last_hidden_state_chosen) # [1024, 1]
                rejected_rewards = model(last_hidden_state_rejected) # [1024, 1]

                # Calculate loss
                padding_chosen = max(0, cutoff_len - len(chosen_input_ids))
                padding_rejected = max(0, cutoff_len - len(rejected_input_ids))
                chosen_input_ids = F.pad(chosen_input_ids, (0, padding_chosen), value = pad_token_id)
                rejected_input_ids = F.pad(rejected_input_ids, (0, padding_rejected), value = pad_token_id)

                chosen_length = (chosen_input_ids != pad_token_id).nonzero()[-1] + 1
                rejected_length = (rejected_input_ids != pad_token_id).nonzero()[-1] + 1
                check_divergence = (chosen_input_ids != rejected_input_ids).nonzero()

                if len(check_divergence) == 0:
                    end_index = chosen_length
                    div_index = end_index - 1
                else:
                    end_index = max(chosen_length, rejected_length)
                    div_index = check_divergence[0]

                chosen_trunc_rewards = chosen_rewards[div_index:end_index]
                rejected_trunc_rewards = rejected_rewards[div_index:end_index]
                loss = -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Accumulate the loss
                
            # Update the learning rate after each epoch
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataset)}, Learning Rate: {scheduler.get_last_lr()}")
        
        # Save parameters if save_params is provided
        if save_path:
            save_file(model.state_dict(), save_path, metadata={"format": "pt"}) # save model
            print(f"Model saved to {save_path}")

    def train(self, percentage = 0.9, num_epochs = 20):
        accelerator = Accelerator()
        device = accelerator.device

        model = self.v_head.to(device) 
        
        cutoff_len = self.data_args.cutoff_len
        pad_token_id = self.tokenizer.pad_token_id

        # training data
        train_dataset = self.get_training_dataset(is_override = self.finetuning_args.is_compute_emb)

        # training args
        # num_epochs = int(self.training_args.num_train_epochs)
        num_training_steps_per_epoch = len(train_dataset) 
        num_training_steps = num_epochs * num_training_steps_per_epoch


        v_head_path = f"{self.training_args.output_dir}/value_head.safetensors"

        # initialize new model and optimizer
        sample_ids = list(range(int(len(train_dataset) * percentage)))

        # optimizer, scheduler
        optimizer_params = self.trainer.create_optimizer().param_groups[0]
        create_scheduler = self.trainer.create_scheduler
        optimizer_params.pop('params', None)     
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        # scheduler = create_scheduler(num_training_steps, optimizer = optimizer)

        model, optimizer, train_dataset = accelerator.prepare(model, optimizer, train_dataset)
        model.train()
            
        # Traing loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            for idx in tqdm(sample_ids):
                example = train_dataset[idx]
                last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
                last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
                chosen_input_ids = example['chosen_ids']
                rejected_input_ids = example['rejected_ids']

                optimizer.zero_grad()
                chosen_rewards = model(last_hidden_state_chosen) # [1024, 1]
                rejected_rewards = model(last_hidden_state_rejected) # [1024, 1]

                # Calculate loss
                padding_chosen = max(0, cutoff_len - len(chosen_input_ids))
                padding_rejected = max(0, cutoff_len - len(rejected_input_ids))
                chosen_input_ids = F.pad(chosen_input_ids, (0, padding_chosen), value = pad_token_id)
                rejected_input_ids = F.pad(rejected_input_ids, (0, padding_rejected), value = pad_token_id)

                chosen_non_zero_indices = (chosen_input_ids != pad_token_id).nonzero()
                rejected_non_zero_indices = (rejected_input_ids != pad_token_id).nonzero()

                if chosen_non_zero_indices.numel() > 0 :
                    chosen_length = chosen_non_zero_indices[-1] + 1
                else:
                    chosen_length = 0

                if rejected_non_zero_indices.numel() > 0:
                    rejected_length = rejected_non_zero_indices[-1] + 1
                else:
                    rejected_length = 0

                check_divergence = (chosen_input_ids != rejected_input_ids).nonzero()

                if len(check_divergence) == 0:
                    end_index = chosen_length
                    div_index = end_index - 1
                else:
                    end_index = max(chosen_length, rejected_length)
                    div_index = check_divergence[0]

                chosen_trunc_rewards = chosen_rewards[div_index:end_index]
                rejected_trunc_rewards = rejected_rewards[div_index:end_index]
                loss = -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Accumulate the loss


            # Update the learning rate after each epoch
            # scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataset)}")
        
        # Save model
        save_file(model.state_dict(), v_head_path, metadata={"format": "pt"}) # save model
        print(f"Model saved to {v_head_path}")

    
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

    def predict_prob(self, question_ids = None):
        # Predict probabilities for given data
        accelerator = Accelerator()
        device = accelerator.device
        
        last_hidden_states, is_load = self.get_embedding(True)
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset, is_load) 

        self.v_head.to(device)
        self.v_head.eval()
        predictions = []
        with torch.no_grad():
            for idx in tqdm(range(len(train_dataset))):
                example = train_dataset[idx]

                last_hidden_state_chosen = example['last_hidden_state_chosen'][-1].to(device)
                last_hidden_state_rejected = example['last_hidden_state_rejected'][-1].to(device)

                chosen_rewards = self.v_head(last_hidden_state_chosen)
                rejected_rewards = self.v_head(last_hidden_state_rejected) 
                rewards_concat = torch.tensor([chosen_rewards, rejected_rewards], device=device)
                
                probs = F.softmax(rewards_concat, dim=0)
                
                pred = {"question_id": example['question_id'],
                        "chosen_rewards": probs[0].item(),  
                        "rejected_rewards": probs[1].item() 
                }

                predictions.append(pred)
    
        return predictions
        
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
                "chosen": [],
                "rejected": []
            }
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    # emb = self.base_model(**batch).logits[:, -1, :] #(bz, ctx, 4096)
                    emb = self.base_model.model(**batch).last_hidden_state #(bz, ctx, 4096)
                    bz, ctx , _ = emb.shape
                    emb = emb.cpu()
                    # find item != 2, set 1
                    mask = torch.zeros((bz, ctx))
                    last_token_emb  = ((batch['input_ids'] != 2).sum(-1) - 1).tolist() 
                    for row, col in enumerate(last_token_emb):
                        mask[row, col] = 1
                    # Mul
                    emb_mul = emb * mask.unsqueeze(-1)
                    # Sum
                    last_emb = emb_mul.sum(1)
                    chosen_emb = [np.array(subarray) for subarray in last_emb[:bz//2]]
                    rejected = [np.array(subarray) for subarray in last_emb[bz//2:]]
                    vector_output["chosen"].extend(chosen_emb)
                    vector_output["rejected"].extend(rejected)

            # Stack 
            vector_output["chosen"] = np.stack(vector_output["chosen"], axis = 0)
            vector_output["rejected"] = np.stack(vector_output["rejected"], axis = 0)

            save_to_pkl(vector_output, f"{filename}.pkl")
            train_df = Dataset.from_dict(vector_output)

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