from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
from scipy.stats import entropy
import random

from sklearn.model_selection import train_test_split
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
    
    def train_oracle(self, emb_dataset, val_size= 0.1, random_state = 0):
        X1 = np.array(emb_dataset['chosen'])
        X2 = np.array(emb_dataset['rejected'])        
        X = X1 - X2
        y = np.zeros(len(X))

        chosen_index = np.random.choice(len(X), size=int(0.5 * len(X)), replace=False)
        for idx in chosen_index:
            X[idx] = -X[idx]
            y[idx] = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=random_state)

        # Fitting the logistic regression model
        model = LogisticRegression(random_state=random_state)
        model.fit(X_train, y_train)

        # Once the model is trained, you can evaluate its performance on the test set
        accuracy = model.score(X_test, y_test)
        print("Accuracy on test set:", accuracy)

        return accuracy
        
    
    def train_eval_oracle(self, nEns, is_compute_emb, val_size = 0.1):
        # Train multiple models and return their weights and average parameter updates
        # training data
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)

        metrics = []
        for m in range(nEns):                        
            print(f"Trainig oracle {m}th ...................")
            val_acc = self.train_oracle(emb_dataset, val_size, random_state=m)

            metrics.append({
                "model_id": m,
                "Accuracy": val_acc,
            })
            print(metrics)

        output_file = f"{self.training_args.output_dir}/committees_{nEns}_oracle_model.json"
        with open(output_file, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

        plot_oracle_acc(metrics, self.training_args.output_dir)
        print(f"Metrics saved to {output_file}")
        
             


