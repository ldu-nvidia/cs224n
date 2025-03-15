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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, GPT2Tokenizer, PreTrainedModel, PretrainedConfig
from peft import get_peft_model, LoraConfig, TaskType, LoHaConfig, LoKrConfig, AdaLoraConfig
from peft.utils import set_peft_model_state_dict
import numpy as np
import os
import torch.nn.functional as F
from models.gpt2 import GPT2Model
import wandb
from random import randint
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from wandbconfigs import get_all_lora_configs
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def compute_metrics(p):
    predictions, labels = p
    # Since the model outputs logits (raw predictions), you need to apply a transformation if necessary
    predictions = predictions.squeeze()  # Flatten the predictions (in case of 2D shape)
    labels = labels.squeeze()  # Flatten the labels as well
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    wandb.log({"mae": mae})
    wandb.log({"mse": mse})
    return {'mse': mse, 'mae': mae}

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
  def __init__(self):
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

def train(dataset, model, tokenizer, peft_config, peft_name, run_name):
    seed_everything(12)
    torch.cuda.empty_cache() 
    
    # Check if the pad_token is not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

    # Tokenizing train, validation, and test sets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    ## select a subset of data to make sure the sweep runs well
    subset_indices_train = [randint(0, 5700) for i in range(2000)]
    subset_indices_test = [randint(0, 1350) for i in range(200)]
    tokenized_datasets['train'] = torch.utils.data.Subset(tokenized_datasets['train'], subset_indices_train)
    tokenized_datasets['test'] = torch.utils.data.Subset(tokenized_datasets['test'], subset_indices_test)
    
    # Apply the right PEFT to model
    if peft_name == "baseline":
      peft_model = model
    else:
      peft_model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=1e-3,
    logging_dir="./logs",
    logging_steps=5,
    load_best_model_at_end=False,
    report_to='wandb',
    greater_is_better=False,
    #metric_for_best_model='mse'
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.eval_dataset = tokenized_datasets["test"]
    trainer.evaluate()

def main():
  all_peft_names = ['lora', 'loha', 'lokr', 'adalora', 'baseline']
  all_model_names = ["BERT", "DistilBERT", "RoBERTa"]
  # load all pretrained models and tokenizers
  device = torch.device('cuda')
  all_models, all_tokenizers = [], []
  all_models.append(BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1).to(device))
  all_tokenizers.append(BertTokenizer.from_pretrained("bert-base-uncased"))
  all_models.append(DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(device))
  all_tokenizers.append(DistilBertTokenizer.from_pretrained("distilbert-base-uncased"))
  all_models.append(RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1).to(device))
  all_tokenizers.append(RobertaTokenizer.from_pretrained("roberta-base"))
  
  dataset = load_dataset("glue", "stsb")
  
  # Start a new W&B run with all models and peft configurations
  for i in range(len(all_models)):
    for j in range(len(all_peft_names)):
      model_name, peft_name = all_model_names[i], all_peft_names[j]
      all_configs = get_all_lora_configs(model_name)
      run_name = "Model: " + model_name + " | PEFT: " + peft_name
      wandb.init(project="Model and PEFT Comparison", name=run_name)
      print(run_name)
      train(dataset, all_models[i], all_tokenizers[i], all_configs[j], all_peft_names[j], run_name)
      wandb.finish()

main()