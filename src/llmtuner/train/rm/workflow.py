# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import gc

import os 
import numpy as np
import random

from ...data import get_dataset, split_dataset
from ...extras.callbacks import FixValueHeadModelCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .collator import PairwiseDataCollatorWithPadding
from .metric import compute_accuracy
from .trainer import PairwiseTrainer, OracleTrainer

from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead

from safetensors import safe_open
from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


from .strategy import LLMStrategy
from .entropy_sampling import EntropySampling
from .random_sampling import RandomSampling
from .committees import QueryByCommittees
from .oracle import Oracle

def run_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


    # only train v_head
    if training_args.do_train:
        for name, param in model.named_parameters():
            if 'v_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        compute_metrics=compute_accuracy,
        **split_dataset(dataset, data_args, training_args),
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        
        # get id question
        question_id = dataset['id']
        trainer.save_predictions(predict_results, question_id)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)



def run_oracle_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    seed = 42,
):
    oracle = Oracle(model_args, data_args, training_args,  finetuning_args, callbacks)

    # Training
    if training_args.do_train: 
        oracle.train_oracle(finetuning_args.num_oracle, finetuning_args.is_compute_emb)
    
    # Evaluation
    if training_args.do_eval:
        oracle.eval_oracle(finetuning_args.is_compute_emb)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_selection(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print(f"Acquisition: {finetuning_args.acquisition}")
    if finetuning_args.acquisition == 'random':
        straytegy = RandomSampling(model_args, data_args, training_args,  finetuning_args, callbacks)
    elif finetuning_args.acquisition == 'max_entropy':
        straytegy = EntropySampling(model_args, data_args, training_args,  finetuning_args, callbacks)
    elif finetuning_args.acquisition == 'qbc':
        straytegy = QueryByCommittees(model_args, data_args, training_args,  finetuning_args, callbacks)
    
    straytegy.query(n = finetuning_args.num_sample_selected, iteration = finetuning_args.active_iter)
    