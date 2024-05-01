from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
from scipy.stats import entropy
import random
from sklearn.linear_model import LogisticRegression


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
        # if not self.finetuning_args.is_compute_emb:
        #     del self.trainer, self.base_model
        pass
    
    def train_oracle(self, emb_dataset, train_ids, v_head_path, model_ith):
        X1 = emb_dataset['chosen']
        X1 = emb_dataset['rejected']        
        Y = 
        LogisticRegression
        

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
        
        # training data
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)
        breakpoint()

        metrics = []
        for m in range(nEns):            
            v_head_path = f"{self.training_args.output_dir}/oracle_{m}.safetensors"

            # emb_dataset_length = len(emb_dataset)
            # sample_ids = random.sample(range(emb_dataset_length), emb_dataset_length)
            # train_ids = sample_ids[:int(emb_dataset_length * percentage)]
            # test_ids = sample_ids[int(emb_dataset_length * percentage):]

            # reinit model    
            
            print(f"Trainig oracle {m}th ...................")
            loss = self.train_oracle(emb_dataset, v_head_path, m)
            # 
            print(f"Eval oracle {m}th ...................")
            # acc = self.evaluate_oracle(model, emb_dataset, test_ids, m, threshold = threshold)

            # metrics.append({
            #     "model_id": m,
            #     "loss": loss,
            #     "Accuracy": acc,
            #     "model_path": v_head_path,
            # })
            # print(metrics)

        output_file = f"{self.training_args.output_dir}/committees_{nEns}_oracle_model.json"
        with open(output_file, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)


        plot_oracle_acc(metrics, self.training_args.output_dir)
        print(f"Metrics saved to {output_file}")
        
             


