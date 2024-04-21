from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import numpy as np
from scipy.stats import entropy

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


    def query(self, n=100, iteration = 0, nEns = 4):
        # Get predictions
        self.train_commitees(nEns)
        self.train_commitees(nEns, True)