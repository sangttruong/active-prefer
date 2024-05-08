from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
from scipy.stats import entropy


from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead

from copy import deepcopy

from torch import nn
import torch
import torch.nn.functional as F

from safetensors.torch import save_file, load_file

from tqdm import tqdm
from collections import Counter


class QueryByCommittees(LLMStrategy):
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        super(QueryByCommittees, self).__init__(model_args, data_args, training_args, finetuning_args, callbacks)

    def _train(self, emb_dataset, val_size=0.1, random_state=0):
        start_time = time.time()

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
        model = LogisticRegression(random_state=random_state, solver='lbfgs', max_iter=500)  # 'sag' solver is efficient for large datasets


        training_start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - training_start_time

        # Once the model is trained, you can evaluate its performance on the test set
        prediction_start_time = time.time()
        accuracy = model.score(X_test, y_test)
        prediction_time = time.time() - prediction_start_time

        print("Accuracy on test set:", accuracy)
        print("Training time:", training_time)
        print("Prediction time:", prediction_time)

        # Save the model to a file
        output_path = f"{self.training_args.output_dir}/logistic_regression_model.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        total_time = time.time() - start_time
        print("Total time:", total_time)

        return accuracy

        
    def _eval(self, emb_testset):
        # Load the model from a file
        model_path = f"{self.finetuning_args.vhead_oracle_path}/logistic_regression_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
        X = np.array(emb_testset['chosen'])
        y = np.ones(len(X)) # chosen = 1

        # Once the model is trained, you can evaluate its performance on the test set
        accuracy = model.score(X, y)
        print("Accuracy on test set:", accuracy)

        return accuracy

    def train_commitees(self, nEns=1, is_continues = False, verbose=False):
        # Train multiple models and return their weights and average parameter updates
        # training data
        emb_dataset = self.get_training_dataset(is_override = is_compute_emb)

        metrics = []
        for m in tqdm(range(nEns)):                        
            print(f"Trainig oracle {m}th ...................")
            val_acc = self._train_oracle(emb_dataset, val_size, random_state=m)

            metrics.append({
                "model_id": m,
                "Accuracy": val_acc,
            })
            print(metrics)

        output_file = f"{self.training_args.output_dir}/committees_{nEns}_oracle_model.json"
        with open(output_file, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

        plot_oracle_acc(metrics, self.training_args.output_dir, self.model_args.model_name_or_path)
        print(f"Metrics saved to {output_file}")
    
    def query_by_commitees(self, n=100, iteration = 0, nEns = 30, theshold = 0.5):
        # Assuming self.training_args.output_dir contains the directory path
        output_dir = self.training_args.output_dir
        # Check if the file exists
        print(f"Update comitees ..................")
        if os.path.exists(os.path.join(output_dir, f"qbc_{nEns-1}.safetensors")):
            save_paths = self.train_commitees(nEns, is_continues = True)
        else:
            save_paths = self.train_commitees(nEns)

        print(f"Voting.............")
        accelerator = Accelerator()
        device = accelerator.device
        
        train_dataset = self.get_training_dataset(is_override = False)

        model = self.v_head.to(device)
        model.eval()

        predictions = {} # id: [scores]
        with torch.no_grad():
            for idx in tqdm(range(len(train_dataset))):
                for model_path in save_paths:
                    vhead_params = load_file(model_path)
                    model.load_state_dict(vhead_params, strict=False)

                    example = train_dataset[idx]
                    question_id = example['question_id']
                    last_hidden_state_chosen = example['last_hidden_state_chosen'][-1].to(device)
                    chosen_rewards = model(last_hidden_state_chosen)
                    probs = F.sigmoid(chosen_rewards)
                    score = 1 if probs.item() > theshold else 0 
                    
                    if question_id not in predictions:
                        predictions[question_id] = [score]
                    else:
                        predictions[question_id].append(score)
        
        votes_entropy = {}
        for question_id, scores in predictions.items():
            votes = list(Counter(scores).values())
            # Calculate softmax of votes
            exp_probs = np.exp(votes)
            softmax_scores = exp_probs / np.sum(exp_probs)
            # Cal entropy
            entropy_value = entropy(softmax_scores, base=2)
            votes_entropy[question_id] = entropy_value

        # Sort questions based on entropy
        sorted_entropy = sorted(votes_entropy.items(), key=lambda x: x[1], reverse=True)
        selected_questions = [question[0] for question in sorted_entropy[:n]]
        
        self.update(question_ids=selected_questions, iteration=iteration)
    
    def query(self, n=100, iteration = 0, nEns = 10):
        # Get predictions
        return self.query_by_commitees(n, iteration, nEns)