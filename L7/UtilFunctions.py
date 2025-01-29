# Utility functions
import torch 
from torch.utils.data import Dataset

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_text = []
        for entry in data:
            instruction_and_input = format_input(entry)
            output = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_and_input + output
            self.encoded_text.append(
                tokenizer.encode(full_text)
            )
    
    def __getitem__(self, index):
        return self.encoded_text[index]

    def __len__(self):
        return len(self.data)


def custom_collate(batch, pad_token_id=50256, ignore_index=-100,
                   allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    ip_list, targets_lst = [], []

    print(f"Max batch length is: {batch_max_length}")

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        ip_list.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(ip_list).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

