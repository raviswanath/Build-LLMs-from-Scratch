import torch
import os 
import sys 
import time
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L6.SpamDataLoader import train_loader, val_loader, test_loader
from L6.GPT2ClassificationRetraining_functions import train_classifier_simple, plot_values, calc_accuracy_loader
from L6.GPT2ClassificationRetraining_functions import classify_review

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

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_Accs, val_accs, ex_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs, eval_freq=50, eval_iter=5
)

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Training completed in {execution_time:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
ex_seen_tensor = torch.linspace(0, ex_seen, len(train_losses))

train_losses = [i.item() for i in train_losses]
val_losses = [i.item() for i in val_losses]
plot_values(epochs_tensor, ex_seen_tensor, train_losses, val_losses)


train_accuracy = calc_accuracy_loader(train_loader, model, device=device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device=device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device=device, num_batches=10)

torch.save(model.state_dict(), "review_classifier.pth")

# print(f"Training Accuracy {train_accuracy:.2f}")
# print(f"Validation Accuracy {val_accuracy:.2f}")
# print(f"Test Accuracy {test_accuracy:.2f}")

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=120
))


text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=120
))