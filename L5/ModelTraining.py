import sys
import os
import torch 
import torch.nn as nn 
import tiktoken

# Add the root directory (parent of L4 and L5) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L4.GPTModel import GPTModel
from L2.Encodings import create_dataloader_v1
import L5.ModelEval as ut 

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

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0004, 
    weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = ut.train_model_simple(model, train_loader, val_loader,
                                                              optimizer, device, num_epochs, eval_freq=5,
                                                              eval_iter=5, start_context="Every effort moves you", 
                                                              tokenizer=tokenizer)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

import numpy as np
epochs_tensor = np.linspace(0, num_epochs, len(train_losses))
train_losses = [i.item() for i in train_losses]
val_losses = [x.item() for x in val_losses]
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)