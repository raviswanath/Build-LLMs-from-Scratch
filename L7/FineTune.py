# Load model directly
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch 
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L7.UtilFunctions import format_input, generate, train_model_simple, calc_loss_loader
from L7.LoadData import train_loader, val_loader, val_data
from L5.ModelEval import text_to_token, token_to_text
import time 

# Load the tokenizer and model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)
# print(model)

torch.manual_seed(123)

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
