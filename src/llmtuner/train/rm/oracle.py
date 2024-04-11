# Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py

from typing import TYPE_CHECKING, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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



if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments




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
    
    
    ########### TRAINING ###############
    np_last_hidden_states = trainer.calculate_last_hidden_state(predict_results, dataset)
    last_hidden_states = torch.tensor(np_last_hidden_states)  # Using torch.tensor()

    accelerator = Accelerator()
    device = accelerator.device

    # Model
    v_head = ValueHead(base_model.config).to(device) # v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
    optimizer = torch.optim.Adam(v_head.parameters())

    # Dataloader
    batch_size = 2
    train_dataset = CustomDataset(last_hidden_states, dataset)  # CustomDataset represents your dataset class
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    v_head, optimizer, data = accelerator.prepare(v_head, optimizer, data)

    v_head.train()
    for epoch in range(2):
        for question_id, last_hidden_state_chosen, last_hidden_state_rejected, chosen_ids, rejected_ids in data_loader:
            # Concate chosen + rejected
            inputs = torch.concat([last_hidden_state_chosen, last_hidden_state_rejected], 0)

            optimizer.zero_grad()

            # Forward
            breakpoint()
            values = model(inputs)
            
            # Split the inputs and rewards into two parts, chosen and rejected
            chosen_input_ids, rejected_input_ids = chosen_ids, rejected_ids
            chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
            chosen_scores, rejected_scores = [], []

            # Loss
            loss = 0
            for i in range(batch_size):
                chosen_length = (chosen_input_ids[i] != tokenizer.pad_token_id).nonzero()[-1] + 1
                rejected_length = (rejected_input_ids[i] != tokenizer.pad_token_id).nonzero()[-1] + 1
                check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

                if len(check_divergence) == 0:
                    end_index = chosen_length
                    div_index = end_index - 1
                else:
                    end_index = max(chosen_length, rejected_length)
                    div_index = check_divergence[0]

                assert div_index > 0
                chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
                rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
                # if return_outputs:  # use the score on the last token except pad token for inference
                    # chosen_scores.append(chosen_rewards[i, chosen_length - 1])
                    # rejected_scores.append(rejected_rewards[i, rejected_length - 1])
                loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

            # Backward
            accelerator.backward(loss)

            optimizer.step()


    ##########################
    del trainer, base_model, v_head
    gc.collect()
    torch.cuda.empty_cache()