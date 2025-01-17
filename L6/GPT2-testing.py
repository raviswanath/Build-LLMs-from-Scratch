import torch
import os 
import sys 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L6.SpamDataLoader import train_loader, val_loader, test_loader

# Load the tokenizer and model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.lm_head = torch.nn.Linear(
    in_features=768,
    out_features=num_classes
)

for param in model.transformer.h[-1].parameters():
    param.requires_grad = True

for param in model.transformer.ln_f.parameters():
    param.requires_grad = True


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.to(device)
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch).logits[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions/num_examples


train_accuracy = calc_accuracy_loader(train_loader, model, device=device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device=device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device=device, num_batches=10)

print(f"Train accuracy {train_accuracy*100:.2f}%")
print(f"Validation accuracy {val_accuracy*100:.2f}%")
print(f"Test accuracy {test_accuracy*100:.2f}%")