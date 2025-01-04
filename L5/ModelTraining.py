import sys
import os
import torch 
import torch.nn as nn 
import tiktoken

# Add the root directory (parent of L4 and L5) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L4.GPTModel import GPTModel
from L2.Encodings import create_dataloader_v1
from L5.ModelEval import calc_loss_loader

torch.manual_seed(123)

# Config dictionary for GPT-2 model
gpt_config = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}



start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(gpt_config)

filepath = 'L2/the-verdict.txt'
with open(filepath, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

# split data into training and validation sets
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=gpt_config["context_length"],
    stride=gpt_config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=gpt_config["context_length"],
    stride=gpt_config["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
