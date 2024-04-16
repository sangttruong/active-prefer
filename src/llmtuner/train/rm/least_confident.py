from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import numpy as np
from scipy.stats import entropy

class LeastConfidence(LLMStrategy):
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        super(LeastConfidence, self).__init__(model_args, data_args, training_args, finetuning_args, callbacks)


    def query(self, n=100):
        # Get predictions
        predictions = self.predict_prob()  # list of predictions

        # Calculate entropy for each question
        chosen_rewards_vals = {}
        for prediction in predictions:
            question_id = prediction['question_id']
            chosen_prob = prediction['chosen_rewards']
            # rejected_prob = prediction['rejected_rewards']
            chosen_rewards_vals[question_id] = chosen_prob

        # Sort questions based on entropy
        sorted_chosen_rewards = sorted(chosen_rewards_vals.items(), key=lambda x: x[1], reverse=False)

        # Select top n questions based on entropy
        selected_questions = [question[0] for question in sorted_chosen_rewards[:n]]
        
        return selected_questions