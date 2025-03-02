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
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    # need input args as class variable since forward need to be changed if peft is true
    self.args = args


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

    #print("using peft in model forward")
    last_token_hidden = self.gpt(input_ids, attention_mask)['last_token']
    #print("last token hidden", last_token_hidden)
    logits = self.gpt.hidden_state_to_token(last_token_hidden)
    #print("logits", logits, logits.shape)
    preds = torch.argmax(logits, dim=1)
    #print("predictions", preds)
    # data sequence length are different, get mean to reduce dimension
    loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), reduction='mean')
    #print("loss", loss)
    return loss, logits

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
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

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    lora_config = LoraConfig(
        r=8,  # Rank for LoRA
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        target_modules=["query", "value"],
        task_type="SEQ_CLS"
    )

    # Apply PEFT (LoRA) to the model
    peft_model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

    # Define a compute metric function
    from sklearn.metrics import accuracy_score

    def compute_metrics(p):
        predictions, labels = p
        preds = predictions.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Start training
    trainer.train()

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--use_peft", type=bool, help='use peft to speed up training', default=False)
  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
