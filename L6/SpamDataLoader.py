import torch 
import os 
import sys 
import pandas as pd
from torch.utils.data import DataLoader

# Add the root directory (parent of L4 and L5) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from L6.PyTorchDataSet_Setup import SpamDataset
import tiktoken 

num_workers = 0
batch_size = 8
torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
train_df = pd.read_csv("L6/train_df.csv")

train_dataset = SpamDataset(
    csv_file="L6/train_df.csv", 
    tokenizer=tokenizer, 
    max_length=None
)

val_dataset = SpamDataset(
    csv_file="L6/val_df.csv",
    tokenizer=tokenizer,
    max_length=None
)

test_dataset = SpamDataset(
    csv_file="L6/test_df.csv",
    tokenizer=tokenizer,
    max_length=None
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=False
)

for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")