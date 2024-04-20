from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import numpy as np
from scipy.stats import entropy
import random

class RandomSampling(LLMStrategy):
    def __init__(
        self, 
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
    ):
        super(RandomSampling, self).__init__(model_args, data_args, training_args, finetuning_args, callbacks)


    def query(self, n=100, iteration = 0):
        # Get predictions
        predictions = self.predict_prob()  # list of predictions

        # Calculate entropy for each question
        entropy_vals = {}
        for prediction in predictions:
            question_id = prediction['question_id']
            chosen_prob = prediction['chosen_rewards']
            rejected_prob = prediction['rejected_rewards']
            softmax_scores = np.array([chosen_prob, rejected_prob])
            entropy_value = entropy(softmax_scores, base=2)
            entropy_vals[question_id] = entropy_value

        questions = list(entropy_vals.keys())
        selected_questions = random.sample(questions, n)

        self.update(question_ids=selected_questions, iteration=iteration)