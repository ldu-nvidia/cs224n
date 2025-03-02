import torch
from datasets import Dataset, load_dataset
print(Dataset)
import argparse
import random
import torch
from mydata import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import set_peft_model_state_dict
import numpy as np
import os
import pickle
from models.gpt2 import GPT2Model

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

    'Takes a batch of sentences and produces embeddings for them.'
    if self.args.use_peft == True:
      assert not labels
      print("using peft in model forward")
      last_token_hidden = self.gpt(input_ids, attention_mask)['last_token']
      print("last token hidden", last_token_hidden)
      logits = self.gpt.hidden_state_to_token(last_token_hidden)
      print("logits", logits)
      preds = torch.argmax(logits, dim=1)
      print("predictions", preds)
      loss = F.cross_entropy(logits, labels, reduction='mean').to(device)
      print("loss", loss)
      return loss, logits
    else: 
      last_token_hidden = self.gpt(input_ids, attention_mask)['last_token']
    # two ways of doing classifications
    # 1. adding a linear layer from hidden to # of classes with extra parameter to learn from
    # 2. use hidden state of last unpadded token to predict what the next token would be: yes/no or even more complex token
    ## no extra classification head needed
    return self.gpt.hidden_state_to_token(last_token_hidden)

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def prepare_data():
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    def convert_data(dataloader, type):
        assert type in ['train', 'dev', 'test'], "type is incorrect"
        file_name = "peft_data_"+type+".pkl"
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            print("file exist!")
            with open(file_name, "rb") as file:
                data = pickle.load(file)
            return data
        else:
            token_ids, attention_mask, labels, sent_ids = [], [], [], []
            for batch in dataloader:
                token_ids.extend(batch['token_ids'].tolist())
                attention_mask.extend(batch['attention_mask'].tolist())
                labels.extend(batch['labels'].tolist())
                sent_ids.extend(batch['sent_ids'])
            # create dictionary with data
            data_dict = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sent_ids': sent_ids,
            }
            peft_data = Dataset.from_dict(data_dict)
            print("pickling datasets")
            with open(file_name, "wb") as file:
                pickle.dump(peft_data, file)
            print("peft data saved successfully")
            return peft_data

    return (convert_data(para_train_dataloader, "train"), convert_data(para_dev_dataloader, 'dev'))

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    train_dataset, dev_dataset = prepare_data()

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    # 3. Load the model
    #model_name = "bert-base-uncased"
    #model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Set up LoRA configuration for PEFT
    lora_config = LoraConfig(
        r=8,  # Rank for LoRA
        lora_alpha=16,
        lora_dropout=0.1,
    )

    # Apply PEFT (LoRA) to the model
    peft_model = get_peft_model(model, lora_config)

    # 5. Set up Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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
