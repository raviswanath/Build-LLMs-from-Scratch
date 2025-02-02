# Load model directly
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch 
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L7.UtilFunctions import format_input, generate, train_model_simple, calc_loss_loader
from L7.LoadData import train_loader, val_loader
from L5.ModelEval import text_to_token, token_to_text

# Load the tokenizer and model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)
# print(model)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
