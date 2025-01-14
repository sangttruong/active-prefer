from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
import random
from scipy.stats import entropy
from tqdm import tqdm
import pickle


class EntropySampling(LLMStrategy):
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        super(EntropySampling, self).__init__(model_args, data_args, training_args, finetuning_args, callbacks)


    def query(self, n=100, is_compute_emb = True, iteration = 0):
        if iteration == 0:
            question_ids = list(set(self.pool_dataset['id']))
            selected_questions = random.sample(question_ids, n)
        elif iteration != 0:
            print(f"Load Selector ..................")
            if os.path.exists(os.path.join(self.training_args.output_dir, f"vhead_0.pkl")):
                model_path = f"{self.training_args.output_dir}/vhead_0.pkl"

            # Get predictions
            print(f"Query ..................")
            emb_dataset = self.get_training_dataset(is_compute_emb)
            predictions = self.predict_prob(emb_dataset, model_path, is_compute_emb) 

            scores_vals = {}
            for question_id, chosen_prob, rejected_prob in tqdm(zip(predictions['question_id'], predictions['chosen_rewards'], predictions['rejected_rewards']), total=len(predictions['question_id'])):
                if question_id in scores_vals:
                    scores_vals[question_id].append(rejected_prob)
                else:
                    scores_vals[question_id] = [chosen_prob, rejected_prob]

            entropy_values = {}
            for question_id, prob_list in scores_vals.items():
                # Calculate softmax scores
                exp_probs = np.exp(prob_list)
                softmax_scores = exp_probs / np.sum(exp_probs)
                
                # Calculate entropy
                entropy_value = entropy(softmax_scores, base=2)
                entropy_values[question_id] = entropy_value

            # Sort questions based on entropy
            sorted_entropy = sorted(entropy_values.items(), key=lambda x: x[1], reverse=True)
            selected_questions = [question[0] for question in sorted_entropy[:n]]
        
        self.update(question_ids=selected_questions, iteration=iteration)