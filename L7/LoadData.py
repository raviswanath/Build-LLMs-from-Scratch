import os 
import sys 
import torch 
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L7.DownloadData import data
from L7.UtilFunctions import format_input, InstructionDataset, custom_collate
from torch.utils.data import DataLoader
import tiktoken 
from functools import partial

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# print(model_input + desired_response)

train_len = int(len(data) * 0.85) 
test_len = int(len(data) * 0.1) 
val_len = train_len - test_len

train_data = data[:train_len]
test_data = data[train_len : train_len + test_len]
val_data = data[train_len + test_len :]

# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))

customized_collate_function = partial(
    custom_collate, 
    device=device, 
    allowed_max_length=1024
)

num_workers = 0
batch_size = 8

train_dataset = InstructionDataset(train_data, tokenizer=tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer=tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_function, 
    shuffle=True,
    drop_last=True, 
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_function, 
    shuffle=True,
    drop_last=False, 
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_function, 
    shuffle=True,
    drop_last=False, 
    num_workers=num_workers
)


# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)