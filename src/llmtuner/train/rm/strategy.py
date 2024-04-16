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

import os 
import numpy as np
import random
import json
import copy

from ...data import get_dataset, split_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .collator import PairwiseDataCollatorWithPadding
from .metric import compute_accuracy
from .trainer import PairwiseTrainer, OracleTrainer
from .workflow import *

from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead



if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

class LLMStrategy:
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        self.tokenizer = load_tokenizer(model_args)
        self.pool_dataset = get_dataset(self.tokenizer, model_args, data_args, training_args, stage="rm")
        self.base_model = load_model(self.tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
        self.data_collator = PairwiseDataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        self.callbacks = callbacks

        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args


        # Replace lm_head with identity
        if hasattr(self.base_model, "lm_head"):
            self.base_model.lm_head = torch.nn.Identity()

        # Update arguments
        training_args.remove_unused_columns = False  # important for pairwise dataset

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
        
        self.v_head = ValueHead(self.base_model.config).to(self.device) 


        if self.data_args.dataset in ['allenai/ai2_arc', 'arc']:
            self.dataset = 'arc_challenge_train'
        elif self.data_args.dataset in ['truthful_qa']:
            self.dataset = 'truthful_qa_train'
        elif self.data_args.dataset in ['Rowan/hellaswag', 'hellaswag']:
            self.dataset = 'hellaswag_train'
        elif self.data_args.dataset in ['winogrande']:
            self.dataset = 'winogrande_train'
        elif self.data_args.dataset in ['cais/mmlu', "mmlu"]:
            self.dataset = 'mmlu_train'
        elif self.data_args.dataset in ['Anthropic/hh-rlhf', "hh-rlhf"]:
            self.dataset = 'hh_rlhf_train'
        elif self.data_args.dataset in ['allenai/reward-bench', "reward-bench", "reward_bench"]:
            self.dataset = 'reward_bench_train'
        else:
            raise(f"Does not support {self.data_args.dataset} dataset yet")

    def query(self, n):
        # Select instances from the pool for labeling
        pass

    def update(self, idxs_lb):
        # Update the labeled indices with newly labeled data
        self.idxs_lb = idxs_lb

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

        if model is None:
            model = self.v_head.to(device) 

        optimizer_params.pop('params', None)     
        optimizer = torch.optim.AdamW(model, **optimizer_params)
        
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
            torch.save({
                'model_state_dict': model.state_dict(),
            }, save_path)

    def train(self, question_ids, seed = 42):
        # Train the model
        last_hidden_states = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset)  # Only need change train_dataset for diff oracle model

        # Select subset for traning by question_ids
        sample_ids = [id for id, example in enumerate(self.pool_dataset) if example['id'] in question_ids]
        
        optimizer_params = self.trainer.create_optimizer().param_groups[0]
        base_model_config = self.base_model.config
        create_scheduler = self.trainer.create_scheduler
        cutoff_len = self.data_args.cutoff_len
        pad_token_id = self.tokenizer.pad_token_id
        seed = seed

        self._train_model(
            train_dataset, 
            cutoff_len, 
            pad_token_id, 
            base_model_config, 
            optimizer_params, 
            create_scheduler,
            self.training_args.num_train_epochs,
            model = None, # default self.v_head
            sample_ids = sample_ids, # importance 
            seed = seed,
        )

    def get_dist(self, nEns=1, verbose=False):
        # Train multiple models and return their weights and average parameter updates
        def weight_reset(layer):
            newLayer = deepcopy(layer)
            if isinstance(layer, nn.Linear):
                newLayer.reset_parameters()
                layer.reset_parameters()

        if verbose: 
            print(' ',flush=True)
            print('training to indicated number of epochs', flush=True)
        
        accelerator = Accelerator()
        device = accelerator.device

        if model is None:
            model = self.v_head.to(device) 

        optimizer_params = self.trainer.create_optimizer().param_groups[0]
        create_scheduler = self.trainer.create_scheduler
        cutoff_len = self.data_args.cutoff_len
        pad_token_id = self.tokenizer.pad_token_id

        # optimizer 
        optimizer_params.pop('params', None)     
        optimizer = torch.optim.AdamW(model, **optimizer_params)
        
        # training data
        last_hidden_states = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset)  
        
        # training args
        num_epochs = int(num_epochs)
        num_training_steps_per_epoch = len(train_dataset) 
        num_training_steps = num_epochs * num_training_steps_per_epoch
        if sample_ids is None:
            sample_ids = list(range(len(train_dataset)))
        
        # scheduler
        scheduler = create_scheduler(num_training_steps, optimizer = optimizer)
        
        model, optimizer, train_dataset = accelerator.prepare(model, optimizer, train_dataset)

        allAvs = []
        allWeights = []

        for m in range(nEns):
            # initialize new model and optimizer
            model =  self.v_head.apply(weight_reset).to(device)
            model.train()

            avIterates = []
            steps = 0 
            k = 0
            ek = (k + 1) * num_training_steps
            pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in model.parameters()])

            # Traing loop
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

                    flat = torch.cat([deepcopy(p.detach().cpu()).flatten() for p in model.parameters()])
                    pVec = pVec + flat
                    steps += 1
                    if steps > ek:
                        avIterates.append(pVec / num_training_steps)
                        pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in model.parameters()])
                        k += 1
                        ek = (k + 1) * num_training_steps

                # Update the learning rate after each epoch
                scheduler.step()

                print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataset)}, Learning Rate: {scheduler.get_last_lr()}")
            
            allAvs.append(avIterates)
            allWeights.append(torch.cat([deepcopy(p.detach().cpu()).flatten() for p in model.parameters()]))
                
        for m in range(nEns):
            avIterates = torch.stack(allAvs[m])
            if k > 1: avIterates = torch.stack(allAvs[m][1:])
            avIterates = avIterates - torch.mean(avIterates, 0)
            allAvs[m] = avIterates

        return allWeights, allAvs

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
        
        last_hidden_states = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset) 

        self.v_head.eval()
        predictions = []
        with torch.no_grad():
            for idx in range(len(train_dataset)):
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
        
        last_hidden_states = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset) 

        self.v_head.eval()
        predictions = []
        with torch.no_grad():
            for idx in range(len(train_dataset)):
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
        
        last_hidden_states = self.get_embedding()
        train_dataset = CustomDataset(last_hidden_states, self.pool_dataset) 

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
    
    def get_embedding(self):
        # Get embeddings from the penultimate layer of the network

        filename = f"{self.training_args.output_dir}/last_hidden_states.npy"
        # Check if the file exists
        if os.path.isfile(filename):
            np_last_hidden_states = np.load(filename)
            print(f"Loaded array from {filename}")
        else:
            predict_results = self.trainer.predict(self.pool_dataset, metric_key_prefix="predict")
            np_last_hidden_states = predict_results.predictions

            # Save the array into a file
            np.save(filename, np_last_hidden_states)
            print(f"Array saved to {filename}")
        
        # Training Oracle model
        last_hidden_states = torch.tensor(np_last_hidden_states)  # Using torch.tensor()

        return last_hidden_states

    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[]):
        # Calculate gradient embeddings for the data
        pass

    # fisher embedding for bait (assumes cross-entropy loss)
    def get_exp_grad_embedding(self, X, Y, probs=[], model=[]):
        # Calculate expected gradient embeddings for the data
        pass

    def update(self, question_ids, iteration = 0):
        output_selected_path = f"{self.data_args.dataset_dir}/selected_entries.json"  
        dataset_info_path = f"{self.data_args.dataset_dir}.datset_info.json"
        ori_dataset_path = f"{self.data_args.dataset_dir}/{self.dataset}.json"

        # Update data training
        select_entries_by_ids(ori_dataset_path, question_ids, output_selected_path)

        with open(dataset_info_path, 'r') as file:
            data_info = json.load(file)
            
            if self.dataset in data_info:
                # append new info to data_infor and store result in json file
                new_data_info = f"{self.dataset}__iter_{iteration}"
                data_info[new_data_info] = copy.deepcopy(data_info[self.dataset])
                data_info[new_data_info]["file_name"] = "selected_entries.json"

                with open(dataset_info_path, 'w') as outfile:
                    json.dump(data_info, outfile, indent=4)

                print("Updated dataset info has been stored in", dataset_info_path)

def select_entries_by_ids(ori_dataset_path, question_ids, output_file):
    selected_entries = []

    # Read data from dataset.json
    with open(ori_dataset_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data and select entries with matching IDs
    remaining_entries = []
    for entry in data:
        if entry["id"] in question_ids:
            selected_entries.append(entry)
        else:
            remaining_entries.append(entry)

    # Write remaining entries back to the original file
    with open(ori_dataset_path, 'w') as file:
        json.dump(remaining_entries, file, indent=4)

    # Write selected entries to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(selected_entries, outfile, indent=4)