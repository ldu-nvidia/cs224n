import torch
from datasets import Dataset, load_dataset
import argparse
import random
from torch import nn
from mydata import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import set_peft_model_state_dict
import numpy as np
import os
import torch.nn.functional as F
from models.gpt2 import GPT2Model
from wbconfig import sweep_configuration
import wandb
from random import randint

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Fix the random seed.
def seed_everything(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""
  def __init__(self, config):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model="gpt2", d=768, l=12, num_heads=12)
    self.paraphrase_detection_head = nn.Linear(768, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  # when using peft, labels need to be provided, else not
  def forward(self, input_ids, attention_mask, labels=None):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """
    last_token_hidden = self.gpt(input_ids, attention_mask)['last_token']
    logits = self.gpt.hidden_state_to_token(last_token_hidden)
    # data sequence length are different, get mean to reduce dimension
    loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), reduction='mean')
    return loss, logits

def train(config):
    seed_everything(3)
    device = torch.device('cuda')
    torch.cuda.empty_cache() 
    dataset = load_dataset("glue", "stsb")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Check if the pad_token is not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

    # Tokenizing train, validation, and test sets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    ## select a subset of data to make sure the sweep runs well
    subset_indices_train = [randint(0, 5700) for i in range(500)]
    subset_indices_test = [randint(0, 1350) for i in range(100)]
    tokenized_datasets['train'] = torch.utils.data.Subset(tokenized_datasets['train'], subset_indices_train)
    tokenized_datasets['test'] = torch.utils.data.Subset(tokenized_datasets['test'], subset_indices_test)
    
    model = ParaphraseGPT(config)
    model = model.to(device)

    lora_config = LoraConfig(
        # sweeping parameter: LORA rank
        r=config.LORA_rank,
        lora_alpha=config.LORA_alpha,
        lora_dropout=config.LORA_dropout,
        use_rslora=True,
        use_dora=config.use_dora, 
        bias='none',
        target_modules=["query", "value"],
        task_type="SEQ_CLS"
    )

    # Apply PEFT (LoRA) to the model
    peft_model = get_peft_model(model, lora_config)
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=config.lr,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=1,
    weight_decay=config.weight_decay,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to='wandb',
    run_name='test1'
    )

    def compute_metrics(p):
        predictions, labels = p
        cross_entropy = F.cross_entropy(torch.tensor(predictions).to(torch.float32), 
        torch.tensor(labels).to(torch.long), reduction='mean')
        wandb.log({"cross entropy": cross_entropy})
        predicted_class = np.argmax(predictions, axis=1)
        return {"cross_entropy": cross_entropy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def main():
  wandb.init(project="peft_gpt2", notes="sweep run for base gpt model on GLUE")
  train(wandb.config)

sweep_configuration = {
  "method": "random",

  "metric": {"goal": "minimize", "name" : "cross entropy"},

  "parameters": {
    "LORA_rank" : {"values" : [1, 2, 4, 8]},
    "LORA_alpha": {"values" : [16, 32, 64]},
    "LORA_dropout" : {"values" : [0.02, 0.1, 0.2]},
    "use_dora" : {"values" : [True, False]},
    "batch_size" : {"values" : [2, 4, 8]},
    "lr" : {"values" : [2e-5, 1e-4, 5e-4]},
    "weight_decay" : {"values" : [1e-3, 1e-2, 1e-1]}
  }
}

sweep_id = wandb.sweep(sweep=sweep_configuration)
wandb.agent(sweep_id, function=main, project="peft_gpt2", count=1000)