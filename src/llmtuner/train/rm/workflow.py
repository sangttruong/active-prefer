# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import gc

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



if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


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

    # breakpoint()
    # only train v_head
    if training_args.do_train and finetuning_args.only_training_vhead:
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
        metrics = trainer.evaluate(metric_key_prefix="eval")
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

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    
class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class CustomDataset(Dataset):
    def __init__(self, embeddings_feature, dataset):
        self.embeddings_feature = embeddings_feature
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        example = self.dataset[i]
        return {"question_id": example['id'], # string 
                "last_hidden_state_chosen": self.embeddings_feature[2*i], # tensor (1024 x 4096)
                "last_hidden_state_rejected": self.embeddings_feature[2*i + 1],  # tensor (1024 x 4096)
                'chosen_ids': torch.tensor(example['chosen_ids']), # list ids
                'rejected_ids': torch.tensor(example['rejected_ids']), # list ids
                }
  
def run_oracle_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    seed = 0,
):
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    base_model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Replace lm_head with identity
    if hasattr(base_model, "lm_head"):
        base_model.lm_head = torch.nn.Identity()

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=base_model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [FixValueHeadModelCallback()],
        compute_metrics=compute_accuracy,
        **split_dataset(dataset, data_args, training_args),
    )

    # Prepare data
    predict_results = trainer.predict(dataset, metric_key_prefix="predict")
    trainer.log_metrics("predict", predict_results.metrics)
    trainer.save_metrics("predict", predict_results.metrics)
    
    
    ##########################
    # Training

    # Save and load
    np_last_hidden_states = predict_results.predictions[:, -1, :]
    last_hidden_states = torch.tensor(np_last_hidden_states)  # Using torch.tensor()

    # Model
    accelerator = Accelerator()
    device = accelerator.device

    # Model
    v_head = ValueHead(base_model.config).to(device) 
    
    # optimizer = torch.optim.AdamW(v_head.parameters(), lr = )
    optimizer_params = trainer.create_optimizer().param_groups[0]
    optimizer_params.pop('params', None)     # Remove 'params' from optimizer parameters
    optimizer = torch.optim.AdamW(v_head.parameters(), **optimizer_params)
    
    breakpoint()
    scheduler = trainer.create_scheduler()(optimizer, step_size=1, gamma=0.9)
    
    # Dataloader
    train_dataset = CustomDataset(last_hidden_states, dataset)  # CustomDataset represents your dataset class
    v_head, optimizer, train_dataset = accelerator.prepare(v_head, optimizer, train_dataset)

    v_head.train()
    for epoch in range(2):
        for example in train_dataset:
            # question_id = example['question_id']
            last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
            last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
            # chosen_ids = example['chosen_ids']
            # rejected_ids = example['rejected_ids']

            # Concate chosen + rejected
            # inputs = torch.concat([last_hidden_state_chosen, last_hidden_state_rejected], 0).to(device)

            optimizer.zero_grad()

            # Forward
            chosen_rewards = v_head(last_hidden_state_chosen)
            rejected_rewards = v_head(last_hidden_state_rejected) 
             
            # Loss
            loss = 0
            loss += -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

            # Backward
            accelerator.backward(loss)

            optimizer.step()

        # Update the learning rate after each epoch
        # scheduler.step()

        # print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")


    ##########################
    del trainer, base_model, v_head
    gc.collect()
    torch.cuda.empty_cache()