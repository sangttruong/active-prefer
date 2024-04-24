from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
from scipy.stats import entropy
import random


from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead

from copy import deepcopy

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


from safetensors.torch import save_file, load_file

from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter
import json


def plot_oracle_acc(metrics, output_dir):
    # Extract accuracy values from the metrics
    accuracies = [metric["Accuracy"] for metric in metrics]

    # Calculate mean and variance of accuracy
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)

    # Plot the image
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=mean_accuracy, color='red', linestyle='--', label=f'Mean: {mean_accuracy:.2f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Accuracies')
    plt.legend()
    plt.grid(True)

    # Annotate mean and variance on the plot
    plt.annotate(f'Mean Accuracy: {mean_accuracy:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
    plt.annotate(f'Variance of Accuracy: {variance_accuracy:.2f}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=12)

    plt.savefig(f'{output_dir}/accuracy_histogram.png')

    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Variance of Accuracy: {variance_accuracy:.2f}")

class Oracle(LLMStrategy):
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        super(Oracle, self).__init__(model_args, data_args, training_args, finetuning_args, callbacks)
        self.oracle_init()

    def oracle_init(self):
        if not self.finetuning_args.is_compute_emb:
            del self.trainer, self.base_model

    def train_oracle(self, model, emb_dataset, train_ids, v_head_path, model_ith):
        accelerator = Accelerator()
        device = accelerator.device

        # training args
        num_epochs = int(self.training_args.num_train_epochs)
        num_training_steps_per_epoch = len(train_ids) 
        num_training_steps = num_epochs * num_training_steps_per_epoch

        optimizer = torch.optim.AdamW(model.parameters(), lr = self.training_args.learning_rate)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        cutoff_len = self.data_args.cutoff_len
        pad_token_id = self.tokenizer.pad_token_id

        model, optimizer, emb_dataset = accelerator.prepare(model, optimizer, emb_dataset)

        # Traing loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Initialize epoch loss
            for idx in tqdm(train_ids):
                example = emb_dataset[idx]
                last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
                last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
                chosen_input_ids = example['chosen_ids']
                rejected_input_ids = example['rejected_ids']

                optimizer.zero_grad()
                chosen_rewards = model(last_hidden_state_chosen) #
                rejected_rewards = model(last_hidden_state_rejected) # 

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
            scheduler.step()

            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_ids)}")
        
        # Save model
        save_file(model.state_dict(), v_head_path, metadata={"format": "pt"}) # save model
        print(f"Model {model_ith} saved to {v_head_path}")

        return epoch_loss / len(train_ids)
        

    def evaluate_oracle(self, model, emb_dataset, test_ids, ith, threshold = 0.5):
        output_dir = self.training_args.output_dir

        print(f"Eval.............")
        accelerator = Accelerator()
        device = accelerator.device

        model = model.to(device)
        model.eval()

        if os.path.exists(os.path.join(output_dir, f"oracle_{ith}.safetensors")):
            vhead_params = load_file(os.path.join(output_dir, f"oracle_{ith}.safetensors"))
            model.load_state_dict(vhead_params, strict=False)

        model, emb_dataset = accelerator.prepare(model, emb_dataset)

        predictions = []
        total_chosens = 0
        with torch.no_grad():
            for idx in tqdm(test_ids):   
                example = emb_dataset[idx]
                question_id = example['question_id']
                last_hidden_state_chosen = example['last_hidden_state_chosen'][-1].to(device)
                
                chosen_rewards = model(last_hidden_state_chosen)

                probs = F.sigmoid(chosen_rewards)
                pred = 1 if probs.item() > threshold else 0 
                total_chosens += pred
                
                predictions.append({
                    "id": question_id,
                    "prob": probs.item(),
                    "chosen": pred
                })

        # Write predictions to a JSON file
        output_file = f"{output_dir}/prediction_oracle_{ith}.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f)
        
        return total_chosens / len(test_ids)

    
    def train_eval_oracle(self, nEns, is_compute_emb, percentage = 0.9, threshold = 0.5):
        # Train multiple models and return their weights and average parameter updates
        def weight_reset(layer):
            newLayer = deepcopy(layer)
            if isinstance(layer, nn.Linear):
                newLayer.reset_parameters()
                layer.reset_parameters()
        
        accelerator = Accelerator()
        device = accelerator.device
        
        # training data
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)
        
        
        metrics = []
        for m in range(nEns):            
            v_head_path = f"{self.training_args.output_dir}/oracle_{m}.safetensors"

            emb_dataset_length = len(emb_dataset)
            sample_ids = random.sample(range(emb_dataset_length), emb_dataset_length)
            train_ids = sample_ids[:int(emb_dataset_length * percentage)]
            test_ids = sample_ids[int(emb_dataset_length * percentage):]

            # reinit model    
            model = self.v_head.apply(weight_reset).to(device)
            
            print(f"Trainig oracle {m}th ...................")
            loss = self.train_oracle(model, emb_dataset, train_ids, v_head_path, m)
            print(f"Eval oracle {m}th ...................")
            acc = self.evaluate_oracle(model, emb_dataset, test_ids, m, threshold = threshold)

            metrics.append({
                "model_id": m,
                "loss": loss,
                "Accuracy": acc,
                "model_path": v_head_path,
            })
            print(metrics)

        output_file = f"{self.training_args.output_dir}/committees_{nEns}_oracle_model.json"
        with open(output_file, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)


        plot_oracle_acc(metrics, self.training_args.output_dir)
        print(f"Metrics saved to {output_file}")
        
             


