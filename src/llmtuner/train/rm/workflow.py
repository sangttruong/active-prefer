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


from .entropy_sampling import EntropySampling

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
    seed = 42,
):
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    base_model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    nearest_multiple = len(dataset) // 8 * 8
    dataset = dataset.select(list(range(nearest_multiple)))

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

    ##########################
    # Training

    # Save and load
    model_name = model_args.model_name_or_path.split('/')[-1]
    dataset_name = data_args.dataset
    filename = f"{training_args.output_dir}/{model_name}/{dataset_name}/last_hidden_states.npy"



    # Check if the file exists
    debug = True
    if not debug and os.path.isfile(filename):
        np_last_hidden_states = np.load(filename)
        print(f"Loaded array from {filename}")
    else:
        # Ensure directory exists or create it
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        np_last_hidden_states = predict_results.predictions

        # Save the array into a file
        np.save(filename, np_last_hidden_states)
        print(f"Array saved to {filename}")
    
    # Training Oracle model
    last_hidden_states = torch.tensor(np_last_hidden_states)  # Using torch.tensor()
    train_dataset = CustomDataset(last_hidden_states, dataset)  # Only need change train_dataset for diff oracle model

    
    optimizer_params = trainer.create_optimizer().param_groups[0]
    base_model_config = base_model.config
    create_scheduler = trainer.create_scheduler
    cutoff_len = data_args.cutoff_len
    pad_token_id = tokenizer.pad_token_id
    percentage = 0.9
    output_vhead = f"{training_args.output_dir}/value_head.safetensors"

    # Training
    train_oracle_model(
        train_dataset, 
        cutoff_len, 
        pad_token_id, 
        base_model_config, 
        optimizer_params, 
        create_scheduler, 
        training_args.num_train_epochs, 
        output_vhead,
        percentage,
        seed,
    )

    # Predict
    if training_args.do_predict:
        pass
    
    ##########################
    del trainer, base_model
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_oracle_model(
    train_dataset, 
    cutoff_len, 
    pad_token_id, 
    base_model_config, 
    optimizer_params, 
    create_scheduler, 
    num_epochs,
    output_vhead,
    percentage=0.9, 
    seed = 42,
):

    set_seed(seed)  # Set seed for reproducibility

    # Model
    accelerator = Accelerator()
    device = accelerator.device
    
    # Model
    v_head = ValueHead(base_model_config).to(device) 
    
    optimizer_params.pop('params', None)     
    optimizer = torch.optim.AdamW(v_head.parameters(), **optimizer_params)
    
    num_epochs = int(num_epochs)
    num_training_steps_per_epoch = int(len(train_dataset) * percentage) 
    num_training_steps = num_epochs * num_training_steps_per_epoch
    sample_ids = random.sample(range(len(train_dataset)), num_training_steps_per_epoch)
    
    scheduler = create_scheduler(num_training_steps, optimizer = optimizer)

    v_head, optimizer, train_dataset = accelerator.prepare(v_head, optimizer, train_dataset)

    v_head.eval()
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize epoch loss
        for idx in sample_ids:
            example = train_dataset[idx]
            last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
            last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
            chosen_input_ids = example['chosen_ids']
            rejected_input_ids = example['rejected_ids']

            optimizer.zero_grad()
            chosen_rewards = v_head(last_hidden_state_chosen) # [1024, 1]
            rejected_rewards = v_head(last_hidden_state_rejected) # [1024, 1]

            # Calculate loss
            padding_chosen = max(0, cutoff_len - len(chosen_input_ids))
            padding_rejected = max(0, cutoff_len - len(rejected_input_ids))
            chosen_input_ids = F.pad(chosen_input_ids, (0, padding_chosen), value = pad_token_id)
            rejected_input_ids = F.pad(rejected_input_ids, (0, padding_rejected), value = pad_token_id)


            chosen_length = (chosen_input_ids != pad_token_id).nonzero()[-1] + 1
            rejected_length = (rejected_input_ids != pad_token_id).nonzero()[-1] + 1
            check_divergence = (chosen_input_ids != rejected_input_ids).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            chosen_trunc_rewards = chosen_rewards[div_index:end_index]
            rejected_trunc_rewards = rejected_rewards[div_index:end_index]
            loss = -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate the loss
            
        # Update the learning rate after each epoch
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataset)}, Learning Rate: {scheduler.get_last_lr()}")

    save_file(v_head.state_dict(), output_vhead, metadata={"format": "pt"}) # save model
    print(f"Model v_head saved to {output_vhead}")

def run_selection(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    
    if finetuning_args.acquisition == 'max_entropy':
        straytegy = EntropySampling(model_args, data_args, training_args,  finetuning_args, callbacks)
    
    straytegy.query(n = 100, iteration = finetuning_args.active_iter)
    