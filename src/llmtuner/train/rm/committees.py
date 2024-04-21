from .strategy import LLMStrategy

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

import numpy as np
from scipy.stats import entropy


from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead

from copy import deepcopy

from torch import nn
import torch
import torch.nn.functional as F

from safetensors.torch import save_file, load_file

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

    def train_commitees(self, nEns=1, is_continues = False, verbose=False):
        # Train multiple models and return their weights and average parameter updates
        def weight_reset(layer):
            newLayer = deepcopy(layer)
            if isinstance(layer, nn.Linear):
                newLayer.reset_parameters()
                layer.reset_parameters()

        if verbose: 
            print(' ',flush=True)
            print('training to indicated number of epochs', flush=True)
        
        accelerator = Accelerator()
        device = accelerator.device

        model = self.v_head.to(device) 
        
        cutoff_len = self.data_args.cutoff_len
        pad_token_id = self.tokenizer.pad_token_id

        # training data
        train_dataset = self.get_training_dataset(is_override = True)

        # training args
        num_epochs = int(self.training_args.num_train_epochs)
        num_training_steps_per_epoch = len(train_dataset) 
        num_training_steps = num_epochs * num_training_steps_per_epoch

        save_paths = []

        for m in range(nEns):
            v_head_path = f"{self.training_args.output_dir}/qbc_{m}.safetensors"

            # initialize new model and optimizer
            sample_ids = list(range(len(train_dataset)))

            # optimizer, scheduler
            optimizer_params = self.trainer.create_optimizer().param_groups[0]
            create_scheduler = self.trainer.create_scheduler
            optimizer_params.pop('params', None)     
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
            scheduler = create_scheduler(num_training_steps, optimizer = optimizer)

            if not is_continues:
                model = self.v_head.apply(weight_reset).to(device)
            else:
                print(f"Continue training from {v_head_path}")
                vhead_params = load_file(v_head_path)
                model.load_state_dict(vhead_params, strict=False)

            model, optimizer, train_dataset = accelerator.prepare(model, optimizer, train_dataset)
            model.train()
             
            # Traing loop
            for epoch in range(num_epochs):
                epoch_loss = 0.0  # Initialize epoch loss
                for idx in sample_ids:
                    example = train_dataset[idx]
                    last_hidden_state_chosen = example['last_hidden_state_chosen'].to(device)
                    last_hidden_state_rejected = example['last_hidden_state_rejected'].to(device)
                    chosen_input_ids = example['chosen_ids']
                    rejected_input_ids = example['rejected_ids']

                    optimizer.zero_grad()
                    chosen_rewards = model(last_hidden_state_chosen) # [1024, 1]
                    rejected_rewards = model(last_hidden_state_rejected) # [1024, 1]

                    # Calculate loss
                    padding_chosen = max(0, cutoff_len - len(chosen_input_ids))
                    padding_rejected = max(0, cutoff_len - len(rejected_input_ids))
                    chosen_input_ids = F.pad(chosen_input_ids, (0, padding_chosen), value = pad_token_id)
                    rejected_input_ids = F.pad(rejected_input_ids, (0, padding_rejected), value = pad_token_id)

                    # chosen_length = (chosen_input_ids != pad_token_id).nonzero()[-1] + 1
                    # rejected_length = (rejected_input_ids != pad_token_id).nonzero()[-1] + 1

                    chosen_non_zero_indices = (chosen_input_ids != pad_token_id).nonzero()
                    rejected_non_zero_indices = (rejected_input_ids != pad_token_id).nonzero()

                    if chosen_non_zero_indices.numel() > 0 :
                        chosen_length = chosen_non_zero_indices[-1] + 1
                    else:
                        chosen_length = 0

                    if rejected_non_zero_indices.numel() > 0:
                        rejected_length = rejected_non_zero_indices[-1] + 1
                    else:
                        rejected_length = 0
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

                print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataset)}")
            
            # Save model
            save_file(model.state_dict(), v_head_path, metadata={"format": "pt"}) # save model
            print(f"Model {m} saved to {v_head_path}")

            save_paths.append(v_head_path)

        return save_paths
    
    

    def query(self, n=100, iteration = 0, nEns = 4):
        # Get predictions
        self.train_commitees(nEns)
        self.train_commitees(nEns, True)