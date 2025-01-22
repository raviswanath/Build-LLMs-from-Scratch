import torch 
import sys 
import os 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L6.GPT2ClassificationRetraining_functions import classify_review


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
# Load the tokenizer and model
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


# read the saved model weights and load into the model object
model_state_dict = torch.load("L6/review_classifier.pth")
model.load_state_dict(model_state_dict)

print(model)

# test model accuracy
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)



print(classify_review(
    text_1, model, tokenizer, device, max_length=120
))

print(classify_review(
    text_2, model, tokenizer, device, max_length=120
))