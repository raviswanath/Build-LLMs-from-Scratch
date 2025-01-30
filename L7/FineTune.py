# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L7.UtilFunctions import format_input
from L7.LoadData import val_data

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)
