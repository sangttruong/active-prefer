from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import os
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from safetensors.torch import save_file, load_file


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


    def query(self, n=100, iteration = 0):
        if iteration != 0:
            print(f"Update comitees ..................")
            if os.path.exists(os.path.join(self.training_args.output_dir, f"value_head.safetensors")):
                v_head_path = f"{self.training_args.output_dir}/value_head.safetensors"
                print(f"Load weight from {v_head_path}")
                vhead_params = load_file(v_head_path)
                self.v_head.load_state_dict(vhead_params, strict=False)

        # Get predictions
        print(f"Query ..................")
        predictions = self.predict_prob()  # list of predictions

        # Calculate entropy for each question
        scores_vals = {}
        for prediction in tqdm(predictions):
            question_id = prediction['question_id']
            chosen_prob = prediction['chosen_rewards']
            rejected_prob = prediction['rejected_rewards']

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