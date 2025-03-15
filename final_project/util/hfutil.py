import importlib
datasets = importlib.import_module('datasets')
from datasets import Dataset as hfDataset
#print(hfDataset)
import os

# convert dataloader into datasets.dataset for hf.trainer class
def convert_loader_to_dataset(dataloader, type):
  assert type in ['train', 'eval', 'test']
  file_name = "peft_data"+type+".pkl"
  if file_name in os.listdir():
    with open(file_name, "rb") as file:
      data = pickle.read(file)
    return data
  else: 
    # iteratre through dataloader and collect data, convert tenso to list
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
    peft_data = hfDataset.from_dict(data_dict)
    print("pickling datasets")
    with open(file_name, "wb") as file:
      pickle.dump(peft_data, file)
    print("peft data saved successfully")
    return peft_data